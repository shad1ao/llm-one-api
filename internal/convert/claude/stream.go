package claude

import (
	"encoding/json"
	"strings"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
	"llm-one-api/internal/convert/sse"
)

// ---- decoder: Claude SSE -> IR ----

type decoder struct {
	started bool
	done    bool
	skip    map[int]bool
	stop    ir.StopReason
	stopSeq string
}

func (Codec) NewStreamDecoder() convert.StreamDecoder { return &decoder{skip: map[int]bool{}} }

type wireStreamEvent struct {
	Type         string          `json:"type"`
	Index        int             `json:"index"`
	Message      *wireResponse   `json:"message"`
	ContentBlock *wireBlock      `json:"content_block"`
	Delta        json.RawMessage `json:"delta"`
	Usage        *wireUsage      `json:"usage"`
	Error        *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type wireDelta struct {
	Type         string `json:"type"`
	Text         string `json:"text"`
	Thinking     string `json:"thinking"`
	Signature    string `json:"signature"`
	PartialJSON  string `json:"partial_json"`
	StopReason   string `json:"stop_reason"`
	StopSequence string `json:"stop_sequence"`
}

func (d *decoder) Feed(ev sse.Event) ([]ir.Event, error) {
	if d.done || ev.Data == "" {
		return nil, nil
	}
	var w wireStreamEvent
	if err := json.Unmarshal([]byte(ev.Data), &w); err != nil {
		return nil, nil // tolerate garbage lines
	}
	if w.Type == "" {
		w.Type = ev.Event
	}
	var out []ir.Event
	switch w.Type {
	case "message_start":
		d.started = true
		e := ir.Event{Type: ir.EventStart}
		if w.Message != nil {
			e.ID, e.Model = w.Message.ID, w.Message.Model
			u := w.Message.Usage.ir()
			e.Usage = &u
		}
		out = append(out, e)
	case "content_block_start":
		b := w.ContentBlock
		if b == nil {
			break
		}
		var p ir.Part
		switch b.Type {
		case "text":
			p = ir.Part{Type: ir.PartText}
		case "thinking":
			p = ir.Part{Type: ir.PartThinking, SigOrigin: self}
		case "redacted_thinking":
			p = ir.Part{Type: ir.PartRedactedThinking, Signature: b.Data, SigOrigin: self}
		case "tool_use":
			p = ir.Part{Type: ir.PartToolCall, ToolCallID: b.ID, ToolName: b.Name}
		default:
			d.skip[w.Index] = true
			return nil, nil
		}
		out = append(out, ir.Event{Type: ir.EventBlockStart, Index: w.Index, Block: &p})
		if b.Type == "text" && b.Text != "" {
			out = append(out, ir.Event{Type: ir.EventDelta, Index: w.Index, Delta: ir.DeltaText, Text: b.Text})
		}
	case "content_block_delta":
		if d.skip[w.Index] {
			break
		}
		var dl wireDelta
		_ = json.Unmarshal(w.Delta, &dl)
		e := ir.Event{Type: ir.EventDelta, Index: w.Index}
		switch dl.Type {
		case "text_delta":
			e.Delta, e.Text = ir.DeltaText, dl.Text
		case "thinking_delta":
			e.Delta, e.Text = ir.DeltaThinking, dl.Thinking
		case "signature_delta":
			e.Delta, e.Text = ir.DeltaSignature, dl.Signature
		case "input_json_delta":
			e.Delta, e.Text = ir.DeltaToolArgs, dl.PartialJSON
		default:
			return nil, nil
		}
		if e.Text != "" {
			out = append(out, e)
		}
	case "content_block_stop":
		if !d.skip[w.Index] {
			out = append(out, ir.Event{Type: ir.EventBlockStop, Index: w.Index})
		}
	case "message_delta":
		var dl wireDelta
		_ = json.Unmarshal(w.Delta, &dl)
		if dl.StopReason != "" {
			d.stop = ir.StopFromClaude(dl.StopReason)
			d.stopSeq = dl.StopSequence
		}
		if w.Usage != nil {
			u := w.Usage.ir()
			out = append(out, ir.Event{Type: ir.EventUsage, Usage: &u})
		}
	case "message_stop":
		d.done = true
		stop := d.stop
		if stop == "" {
			stop = ir.StopEndTurn
		}
		out = append(out, ir.Event{Type: ir.EventStop, StopReason: stop, StopSeq: d.stopSeq})
	case "error":
		d.done = true
		e := &ir.APIError{Status: 500, Type: "api_error", Message: "upstream stream error"}
		if w.Error != nil {
			e.Type, e.Message = w.Error.Type, w.Error.Message
			if e.Type == "overloaded_error" {
				e.Status = 529
			}
		}
		out = append(out, ir.Event{Type: ir.EventError, Err: e})
	}
	return out, nil
}

func (d *decoder) Finish() ([]ir.Event, error) {
	if d.done {
		return nil, nil
	}
	if d.stop != "" {
		// message_delta with stop_reason arrived; treat a missing message_stop as complete.
		d.done = true
		return []ir.Event{{Type: ir.EventStop, StopReason: d.stop, StopSeq: d.stopSeq}}, nil
	}
	return nil, convert.ErrIncompleteStream
}

// ---- encoder: IR -> Claude SSE ----

type blockState struct {
	wire    int
	part    ir.Part
	sig     strings.Builder
	hasArgs bool
}

type encoder struct {
	model    string
	id       string
	started  bool
	finished bool
	nextWire int
	open     int // ir index of the open wire block, -1 if none
	blocks   map[int]*blockState
	closed   map[int]bool
	usage    ir.Usage
	sawTool  bool
}

func (Codec) NewStreamEncoder(req *ir.Request) convert.StreamEncoder {
	e := &encoder{open: -1, blocks: map[int]*blockState{}, closed: map[int]bool{}}
	if req != nil {
		e.model = req.Model
	}
	return e
}

func event(name string, payload obj) sse.Event {
	payload["type"] = name
	b, _ := json.Marshal(payload)
	return sse.Event{Event: name, Data: string(b)}
}

func (e *encoder) start(id, model string) []sse.Event {
	if e.started {
		return nil
	}
	e.started = true
	if model != "" && e.model == "" {
		e.model = model
	}
	e.id = id
	if !strings.HasPrefix(e.id, "msg_") {
		e.id = ir.NewID("msg_")
	}
	return []sse.Event{event("message_start", obj{"message": obj{
		"id": e.id, "type": "message", "role": "assistant", "model": e.model,
		"content": []any{}, "stop_reason": nil, "stop_sequence": nil, "usage": usageObj(e.usage),
	}})}
}

func (e *encoder) closeOpen() []sse.Event {
	if e.open < 0 {
		return nil
	}
	idx := e.open
	b := e.blocks[idx]
	e.open = -1
	e.closed[idx] = true
	var out []sse.Event
	switch b.part.Type {
	case ir.PartThinking:
		sig := ir.SigFor(self, b.part.SigOrigin, b.sig.String())
		if sig != "" {
			out = append(out, event("content_block_delta", obj{"index": b.wire,
				"delta": obj{"type": "signature_delta", "signature": sig}}))
		}
	case ir.PartToolCall:
		if !b.hasArgs {
			out = append(out, event("content_block_delta", obj{"index": b.wire,
				"delta": obj{"type": "input_json_delta", "partial_json": "{}"}}))
		}
	}
	return append(out, event("content_block_stop", obj{"index": b.wire}))
}

func (e *encoder) Encode(ev ir.Event) []sse.Event {
	if e.finished {
		return nil
	}
	var out []sse.Event
	if ev.Type != ir.EventStart && ev.Type != ir.EventError && ev.Type != ir.EventUsage {
		out = append(out, e.start("", "")...)
	}
	switch ev.Type {
	case ir.EventStart:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
		out = append(out, e.start(ev.ID, ev.Model)...)
	case ir.EventBlockStart:
		p := *ev.Block
		var cb obj
		switch p.Type {
		case ir.PartText:
			cb = obj{"type": "text", "text": ""}
		case ir.PartThinking:
			cb = obj{"type": "thinking", "thinking": "", "signature": ""}
		case ir.PartRedactedThinking:
			if p.SigOrigin == self {
				cb = obj{"type": "redacted_thinking", "data": p.Signature}
			} else {
				p.Type = ir.PartThinking // foreign opaque reasoning, keep replayable
				cb = obj{"type": "thinking", "thinking": "", "signature": ""}
			}
		case ir.PartToolCall:
			e.sawTool = true
			cb = obj{"type": "tool_use", "id": clientToolID(p), "name": p.ToolName, "input": obj{}}
		default:
			return out
		}
		out = append(out, e.closeOpen()...)
		b := &blockState{wire: e.nextWire, part: p}
		if p.Type == ir.PartThinking && ev.Block.Type == ir.PartRedactedThinking {
			b.sig.WriteString(ev.Block.Signature)
		}
		e.nextWire++
		e.blocks[ev.Index] = b
		e.open = ev.Index
		out = append(out, event("content_block_start", obj{"index": b.wire, "content_block": cb}))
	case ir.EventDelta:
		b, ok := e.blocks[ev.Index]
		if !ok || e.closed[ev.Index] || e.open != ev.Index {
			return out // late delta for a closed block; cannot be expressed in Claude SSE
		}
		var d obj
		switch ev.Delta {
		case ir.DeltaText:
			d = obj{"type": "text_delta", "text": ev.Text}
		case ir.DeltaThinking:
			d = obj{"type": "thinking_delta", "thinking": ev.Text}
		case ir.DeltaSignature:
			b.sig.WriteString(ev.Text)
			return out
		case ir.DeltaToolArgs:
			b.hasArgs = true
			d = obj{"type": "input_json_delta", "partial_json": ev.Text}
		}
		out = append(out, event("content_block_delta", obj{"index": b.wire, "delta": d}))
	case ir.EventBlockStop:
		if e.open == ev.Index {
			out = append(out, e.closeOpen()...)
		}
	case ir.EventUsage:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
	case ir.EventStop:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
		stop := ev.StopReason
		if stop == ir.StopEndTurn && e.sawTool {
			stop = ir.StopToolUse
		}
		out = append(out, e.finish(stop, ev.StopSeq)...)
	case ir.EventError:
		e.finished = true
		err := ev.Err
		if err == nil {
			err = &ir.APIError{Status: 500, Type: "api_error", Message: "stream error"}
		}
		out = append(out, event("error", obj{"error": obj{"type": claudeErrType(err), "message": err.Message}}))
	}
	return out
}

func (e *encoder) finish(stop ir.StopReason, stopSeq string) []sse.Event {
	out := e.start("", "")
	out = append(out, e.closeOpen()...)
	var seq any
	if stopSeq != "" {
		seq = stopSeq
	}
	out = append(out,
		event("message_delta", obj{"delta": obj{"stop_reason": ir.StopToClaude(stop), "stop_sequence": seq}, "usage": usageObj(e.usage)}),
		event("message_stop", obj{}))
	e.finished = true
	return out
}

func (e *encoder) Finalize() []sse.Event {
	if e.finished {
		return nil
	}
	stop := ir.StopEndTurn
	if e.sawTool {
		stop = ir.StopToolUse
	}
	return e.finish(stop, "")
}
