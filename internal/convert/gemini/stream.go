package gemini

import (
	"encoding/json"
	"strings"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
	"llm-one-api/internal/convert/sse"
)

// ---- decoder: Gemini SSE (alt=sse) -> IR ----

type decoder struct {
	started  bool
	done     bool
	next     int
	open     int
	openType ir.PartType
	sawTool  bool
	idGen    func() string
}

func (Codec) NewStreamDecoder() convert.StreamDecoder {
	return &decoder{open: -1, idGen: newIDGen()}
}

func (d *decoder) closeOpen() []ir.Event {
	if d.open < 0 {
		return nil
	}
	idx := d.open
	d.open = -1
	return []ir.Event{{Type: ir.EventBlockStop, Index: idx}}
}

func (d *decoder) ensure(t ir.PartType) []ir.Event {
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

func (d *decoder) Feed(ev sse.Event) ([]ir.Event, error) {
	data := strings.TrimSpace(ev.Data)
	if d.done || data == "" {
		return nil, nil
	}
	var w wireResponse
	if err := json.Unmarshal([]byte(data), &w); err != nil {
		return nil, nil
	}
	if w.Error != nil {
		d.done = true
		return []ir.Event{{Type: ir.EventError, Err: &ir.APIError{Status: max(w.Error.Code, 500), Type: "api_error", Message: w.Error.Message}}}, nil
	}
	var out []ir.Event
	if !d.started {
		d.started = true
		out = append(out, ir.Event{Type: ir.EventStart, ID: w.ResponseID, Model: w.ModelVersion})
	}
	finish := ""
	if len(w.Candidates) > 0 {
		c := w.Candidates[0]
		if c.Content != nil {
			for _, p := range partsFrom(c.Content.Parts, d.idGen) {
				out = append(out, d.part(p)...)
			}
			// A signature on the trailing thought part arrives with it.
		}
		finish = c.FinishReason
	} else if w.PromptFeedback != nil && w.PromptFeedback.BlockReason != "" {
		finish = "SAFETY"
	}
	if w.UsageMetadata != nil {
		u := w.UsageMetadata.ir()
		out = append(out, ir.Event{Type: ir.EventUsage, Usage: &u})
	}
	if finish != "" {
		d.done = true
		out = append(out, d.closeOpen()...)
		stop := ir.StopFromGemini(finish)
		if stop == ir.StopEndTurn && d.sawTool {
			stop = ir.StopToolUse
		}
		out = append(out, ir.Event{Type: ir.EventStop, StopReason: stop})
	}
	return out, nil
}

func (d *decoder) part(p ir.Part) []ir.Event {
	switch p.Type {
	case ir.PartText:
		out := d.ensure(ir.PartText)
		return append(out, ir.Event{Type: ir.EventDelta, Index: d.open, Delta: ir.DeltaText, Text: p.Text})
	case ir.PartThinking:
		out := d.ensure(ir.PartThinking)
		if p.Text != "" {
			out = append(out, ir.Event{Type: ir.EventDelta, Index: d.open, Delta: ir.DeltaThinking, Text: p.Text})
		}
		if p.Signature != "" {
			out = append(out, ir.Event{Type: ir.EventDelta, Index: d.open, Delta: ir.DeltaSignature, Text: p.Signature})
		}
		return out
	case ir.PartToolCall:
		d.sawTool = true
		out := d.closeOpen()
		idx := d.next
		d.next++
		block := p
		block.Arguments = ""
		return append(out,
			ir.Event{Type: ir.EventBlockStart, Index: idx, Block: &block},
			ir.Event{Type: ir.EventDelta, Index: idx, Delta: ir.DeltaToolArgs, Text: p.Arguments},
			ir.Event{Type: ir.EventBlockStop, Index: idx})
	}
	return nil
}

func (d *decoder) Finish() ([]ir.Event, error) {
	if d.done {
		return nil, nil
	}
	return nil, convert.ErrIncompleteStream
}

// ---- encoder: IR -> Gemini SSE ----

type pendingCall struct {
	part ir.Part
	args strings.Builder
}

type encoder struct {
	id       string
	model    string
	started  bool
	finished bool
	calls    map[int]*pendingCall
	sigs     map[int]*strings.Builder
	thinking map[int]ir.Part
	usage    ir.Usage
	sawTool  bool
}

func (Codec) NewStreamEncoder(req *ir.Request) convert.StreamEncoder {
	e := &encoder{calls: map[int]*pendingCall{}, sigs: map[int]*strings.Builder{}, thinking: map[int]ir.Part{}}
	if req != nil {
		e.model = req.Model
	}
	return e
}

func (e *encoder) chunk(parts []obj, finish string, usage bool) sse.Event {
	cand := obj{"content": obj{"role": "model", "parts": parts}, "index": 0}
	if finish != "" {
		cand["finishReason"] = finish
	}
	o := obj{"candidates": []obj{cand}, "modelVersion": e.model, "responseId": e.id}
	if usage {
		o["usageMetadata"] = usageObj(e.usage)
	}
	b, _ := json.Marshal(o)
	return sse.Event{Data: string(b)}
}

func (e *encoder) Encode(ev ir.Event) []sse.Event {
	if e.finished {
		return nil
	}
	if !e.started && ev.Type != ir.EventError {
		e.started = true
		e.id = firstNonEmpty(ev.ID, ir.NewID(""))
		if e.model == "" {
			e.model = ev.Model
		}
	}
	switch ev.Type {
	case ir.EventStart, ir.EventUsage:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
	case ir.EventBlockStart:
		switch ev.Block.Type {
		case ir.PartToolCall:
			e.sawTool = true
			e.calls[ev.Index] = &pendingCall{part: *ev.Block}
		case ir.PartThinking:
			e.thinking[ev.Index] = *ev.Block
			e.sigs[ev.Index] = &strings.Builder{}
		}
	case ir.EventDelta:
		switch ev.Delta {
		case ir.DeltaText:
			return []sse.Event{e.chunk([]obj{{"text": ev.Text}}, "", false)}
		case ir.DeltaThinking:
			return []sse.Event{e.chunk([]obj{{"text": ev.Text, "thought": true}}, "", false)}
		case ir.DeltaSignature:
			if b := e.sigs[ev.Index]; b != nil {
				b.WriteString(ev.Text)
			}
		case ir.DeltaToolArgs:
			if c := e.calls[ev.Index]; c != nil {
				c.args.WriteString(ev.Text)
			}
		}
	case ir.EventBlockStop:
		if c := e.calls[ev.Index]; c != nil {
			delete(e.calls, ev.Index)
			p := c.part
			p.Arguments = c.args.String()
			return []sse.Event{e.chunk(clientParts([]ir.Part{p}), "", false)}
		}
		if b := e.sigs[ev.Index]; b != nil {
			delete(e.sigs, ev.Index)
			p := e.thinking[ev.Index]
			if s := sigToClient(p.SigOrigin, b.String()); s != "" {
				return []sse.Event{e.chunk([]obj{{"text": "", "thought": true, "thoughtSignature": s}}, "", false)}
			}
		}
	case ir.EventStop:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
		stop := ev.StopReason
		return e.finish(stop)
	case ir.EventError:
		e.finished = true
		err := ev.Err
		if err == nil {
			err = &ir.APIError{Status: 500, Message: "stream error"}
		}
		b, _ := json.Marshal(obj{"error": errorObj(err)})
		return []sse.Event{{Data: string(b)}}
	}
	return nil
}

func (e *encoder) finish(stop ir.StopReason) []sse.Event {
	var out []sse.Event
	for idx, c := range e.calls {
		delete(e.calls, idx)
		p := c.part
		p.Arguments = c.args.String()
		out = append(out, e.chunk(clientParts([]ir.Part{p}), "", false))
	}
	out = append(out, e.chunk([]obj{{"text": ""}}, finishReason(stop), true))
	e.finished = true
	return out
}

func (e *encoder) Finalize() []sse.Event {
	if e.finished {
		return nil
	}
	if !e.started {
		e.started = true
		e.id = ir.NewID("")
	}
	return e.finish(ir.StopEndTurn)
}
