package openaichat

import (
	"encoding/json"
	"strings"
	"time"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
	"llm-one-api/internal/convert/sse"
)

// ---- decoder: chat.completion.chunk SSE -> IR ----

type toolState struct {
	irIdx   int
	id      string
	name    string
	pending strings.Builder // args received before id+name were known
	started bool
}

type decoder struct {
	started  bool
	done     bool
	finished bool // finish_reason seen
	stop     ir.StopReason
	next     int
	open     int // ir index of open block, -1 none
	openType ir.PartType
	byIndex  map[int]*toolState
	order    []*toolState
	last     *toolState
	sawTool  bool
}

func (Codec) NewStreamDecoder() convert.StreamDecoder {
	return &decoder{open: -1, byIndex: map[int]*toolState{}}
}

func (d *decoder) closeOpen() []ir.Event {
	if d.open < 0 {
		return nil
	}
	idx := d.open
	d.open = -1
	return []ir.Event{{Type: ir.EventBlockStop, Index: idx}}
}

// ensureBlock makes sure a text/thinking block is open.
func (d *decoder) ensureBlock(t ir.PartType) []ir.Event {
	if d.open >= 0 && d.openType == t {
		return nil
	}
	out := d.closeOpen()
	d.open, d.openType = d.next, t
	d.next++
	p := ir.Part{Type: t}
	if t == ir.PartThinking {
		p.SigOrigin = self
	}
	return append(out, ir.Event{Type: ir.EventBlockStart, Index: d.open, Block: &p})
}

func (d *decoder) startTool(ts *toolState) []ir.Event {
	out := d.closeOpen()
	if ts.id == "" {
		ts.id = ir.NewID("call_")
	}
	ts.irIdx = d.next
	d.next++
	ts.started = true
	d.sawTool = true
	d.open, d.openType = ts.irIdx, ir.PartToolCall
	id, origin, sig := ir.UnpackCallID(ts.id)
	out = append(out, ir.Event{Type: ir.EventBlockStart, Index: ts.irIdx,
		Block: &ir.Part{Type: ir.PartToolCall, ToolCallID: id, ToolName: ts.name, Signature: sig, SigOrigin: origin}})
	if ts.pending.Len() > 0 {
		out = append(out, ir.Event{Type: ir.EventDelta, Index: ts.irIdx, Delta: ir.DeltaToolArgs, Text: ts.pending.String()})
		ts.pending.Reset()
	}
	return out
}

func (d *decoder) Feed(ev sse.Event) ([]ir.Event, error) {
	data := strings.TrimSpace(ev.Data)
	if d.done || data == "" {
		return nil, nil
	}
	if data == "[DONE]" {
		return d.end(), nil
	}
	var w wireResponse
	if err := json.Unmarshal([]byte(data), &w); err != nil {
		return nil, nil
	}
	var out []ir.Event
	if w.Error != nil && w.Error.Message != "" {
		d.done = true
		return []ir.Event{{Type: ir.EventError, Err: &ir.APIError{Status: 500, Type: firstNonEmpty(w.Error.Type, "api_error"), Message: w.Error.Message}}}, nil
	}
	if !d.started {
		d.started = true
		out = append(out, ir.Event{Type: ir.EventStart, ID: w.ID, Model: w.Model})
	}
	for _, c := range w.Choices {
		if c.Index != 0 {
			continue
		}
		if dl := c.Delta; dl != nil {
			if rc := firstNonEmpty(dl.ReasoningContent, dl.Reasoning); rc != "" {
				out = append(out, d.ensureBlock(ir.PartThinking)...)
				out = append(out, ir.Event{Type: ir.EventDelta, Index: d.open, Delta: ir.DeltaThinking, Text: rc})
			}
			if s, ok := ir.TextOrParts(dl.Content); ok && s != "" {
				out = append(out, d.ensureBlock(ir.PartText)...)
				out = append(out, ir.Event{Type: ir.EventDelta, Index: d.open, Delta: ir.DeltaText, Text: s})
			}
			for _, tc := range dl.ToolCalls {
				out = append(out, d.toolDelta(tc)...)
			}
		}
		if c.FinishReason != "" {
			d.finished = true
			d.stop = ir.StopFromOpenAI(c.FinishReason)
			out = append(out, d.flushTools()...)
			out = append(out, d.closeOpen()...)
		}
	}
	if w.Usage != nil && (w.Usage.PromptTokens > 0 || w.Usage.CompletionTokens > 0) {
		u := w.Usage.ir()
		out = append(out, ir.Event{Type: ir.EventUsage, Usage: &u})
	}
	return out, nil
}

func (d *decoder) toolDelta(tc wireToolCall) []ir.Event {
	var ts *toolState
	if tc.Index != nil {
		ts = d.byIndex[*tc.Index]
	} else if tc.ID == "" {
		ts = d.last // index-less continuation (some compat providers)
	}
	// A different id at a known index means a new call reusing the slot.
	if ts != nil && tc.ID != "" && ts.id != "" && tc.ID != ts.id {
		ts = nil
	}
	if ts == nil {
		ts = &toolState{}
		d.order = append(d.order, ts)
		if tc.Index != nil {
			d.byIndex[*tc.Index] = ts
		}
	}
	d.last = ts
	if tc.ID != "" && ts.id == "" {
		ts.id = tc.ID
	}
	if tc.Function.Name != "" && ts.name == "" {
		ts.name = tc.Function.Name // some providers repeat the name on every chunk; take the first
	}
	var out []ir.Event
	if !ts.started && ts.id != "" && ts.name != "" {
		out = append(out, d.startTool(ts)...)
	}
	if args := tc.Function.Arguments; args != "" {
		if ts.started {
			out = append(out, ir.Event{Type: ir.EventDelta, Index: ts.irIdx, Delta: ir.DeltaToolArgs, Text: args})
		} else {
			ts.pending.WriteString(args)
		}
	}
	return out
}

// flushTools starts tool calls that never received an id (synthesizing one).
func (d *decoder) flushTools() []ir.Event {
	var out []ir.Event
	for _, ts := range d.order {
		if !ts.started && ts.name != "" {
			out = append(out, d.startTool(ts)...)
		}
	}
	return out
}

func (d *decoder) end() []ir.Event {
	d.done = true
	out := d.flushTools()
	out = append(out, d.closeOpen()...)
	if !d.started {
		out = append(out, ir.Event{Type: ir.EventStart})
	}
	stop := d.stop
	if stop == "" || (stop == ir.StopEndTurn && d.sawTool) {
		if d.sawTool {
			stop = ir.StopToolUse
		} else {
			stop = ir.StopEndTurn
		}
	}
	return append(out, ir.Event{Type: ir.EventStop, StopReason: stop})
}

func (d *decoder) Finish() ([]ir.Event, error) {
	if d.done {
		return nil, nil
	}
	if d.finished {
		return d.end(), nil
	}
	return nil, convert.ErrIncompleteStream
}

// ---- encoder: IR -> chat.completion.chunk SSE ----

type encoder struct {
	id           string
	model        string
	created      int64
	includeUsage bool
	started      bool
	finished     bool
	toolIdx      map[int]int
	tools        int
	usage        ir.Usage
	hasUsage     bool
}

func (Codec) NewStreamEncoder(req *ir.Request) convert.StreamEncoder {
	e := &encoder{created: time.Now().Unix(), toolIdx: map[int]int{}}
	if req != nil {
		e.model = req.Model
		e.includeUsage = req.IncludeUsage
	}
	return e
}

func (e *encoder) chunk(delta obj, finish any) sse.Event {
	b, _ := json.Marshal(obj{
		"id": e.id, "object": "chat.completion.chunk", "created": e.created, "model": e.model,
		"choices": []obj{{"index": 0, "delta": delta, "finish_reason": finish}},
	})
	return sse.Event{Data: string(b)}
}

func (e *encoder) start(id, model string) []sse.Event {
	if e.started {
		return nil
	}
	e.started = true
	e.id = chatID(id)
	if e.model == "" {
		e.model = model
	}
	return []sse.Event{e.chunk(obj{"role": "assistant", "content": ""}, nil)}
}

func (e *encoder) Encode(ev ir.Event) []sse.Event {
	if e.finished {
		return nil
	}
	var out []sse.Event
	if ev.Type != ir.EventError && ev.Type != ir.EventUsage {
		out = append(out, e.start(ev.ID, ev.Model)...)
	}
	switch ev.Type {
	case ir.EventBlockStart:
		if ev.Block.Type == ir.PartToolCall {
			n := e.tools
			e.tools++
			e.toolIdx[ev.Index] = n
			out = append(out, e.chunk(obj{"tool_calls": []obj{{"index": n, "id": clientToolID(*ev.Block), "type": "function",
				"function": obj{"name": ev.Block.ToolName, "arguments": ""}}}}, nil))
		}
	case ir.EventDelta:
		switch ev.Delta {
		case ir.DeltaText:
			out = append(out, e.chunk(obj{"content": ev.Text}, nil))
		case ir.DeltaThinking:
			out = append(out, e.chunk(obj{"reasoning_content": ev.Text}, nil))
		case ir.DeltaToolArgs:
			if n, ok := e.toolIdx[ev.Index]; ok {
				out = append(out, e.chunk(obj{"tool_calls": []obj{{"index": n, "function": obj{"arguments": ev.Text}}}}, nil))
			}
		}
	case ir.EventUsage:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
			e.hasUsage = true
		}
	case ir.EventStart:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
	case ir.EventStop:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
			e.hasUsage = true
		}
		out = append(out, e.finish(ev.StopReason)...)
	case ir.EventError:
		e.finished = true
		err := ev.Err
		if err == nil {
			err = &ir.APIError{Status: 500, Type: "api_error", Message: "stream error"}
		}
		b, _ := json.Marshal(obj{"error": errorObj(err)})
		out = append(out, sse.Event{Data: string(b)})
	}
	return out
}

func (e *encoder) finish(stop ir.StopReason) []sse.Event {
	out := e.start("", "")
	if stop == ir.StopEndTurn && e.tools > 0 {
		stop = ir.StopToolUse
	}
	out = append(out, e.chunk(obj{}, ir.StopToOpenAI(stop)))
	if e.includeUsage {
		b, _ := json.Marshal(obj{"id": e.id, "object": "chat.completion.chunk", "created": e.created, "model": e.model,
			"choices": []obj{}, "usage": usageObj(e.usage)})
		out = append(out, sse.Event{Data: string(b)})
	}
	out = append(out, sse.Event{Data: "[DONE]"})
	e.finished = true
	return out
}

func (e *encoder) Finalize() []sse.Event {
	if e.finished {
		return nil
	}
	return e.finish(ir.StopEndTurn)
}
