package openairesponses

import (
	"encoding/json"
	"strings"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
	"llm-one-api/internal/convert/sse"
)

// ---- decoder: Responses SSE -> IR ----

type itemState struct {
	irIdx     int // block for reasoning / function_call; -1 if none
	typ       string
	callID    string
	args      strings.Builder // streamed arguments (to compute .done remainder)
	custom    bool
	summaryIx int
	textIdx   map[int]int // message items: content_index -> ir index
	hadText   bool        // any text was streamed (or recovered) for this item
}

type decoder struct {
	started bool
	done    bool
	next    int
	items   map[string]*itemState
	byOut   map[int]*itemState
	sawTool bool
}

func (Codec) NewStreamDecoder() convert.StreamDecoder {
	return &decoder{items: map[string]*itemState{}, byOut: map[int]*itemState{}}
}

type wireEvent struct {
	Type         string        `json:"type"`
	Response     *wireResponse `json:"response"`
	Item         *wireItem     `json:"item"`
	ItemID       string        `json:"item_id"`
	OutputIndex  int           `json:"output_index"`
	ContentIndex int           `json:"content_index"`
	SummaryIndex int           `json:"summary_index"`
	Delta        string        `json:"delta"`
	Arguments    string        `json:"arguments"`
	Input        string        `json:"input"`
	Text         string        `json:"text"`
	Code         string        `json:"code"`
	Message      string        `json:"message"`
}

func (d *decoder) item(ev *wireEvent) *itemState {
	if ev.ItemID != "" {
		if it := d.items[ev.ItemID]; it != nil {
			return it
		}
	}
	return d.byOut[ev.OutputIndex]
}

func (d *decoder) block(p ir.Part) (int, ir.Event) {
	idx := d.next
	d.next++
	return idx, ir.Event{Type: ir.EventBlockStart, Index: idx, Block: &p}
}

// jsonFragment escapes s as the inside of a JSON string literal.
func jsonFragment(s string) string {
	b, _ := json.Marshal(s)
	return string(b[1 : len(b)-1])
}

func (d *decoder) Feed(sev sse.Event) ([]ir.Event, error) {
	if d.done || sev.Data == "" || sev.Data == "[DONE]" {
		return nil, nil
	}
	var ev wireEvent
	if err := json.Unmarshal([]byte(sev.Data), &ev); err != nil {
		return nil, nil
	}
	if ev.Type == "" {
		ev.Type = sev.Event
	}
	var out []ir.Event
	if !d.started && ev.Type != "error" {
		d.started = true
		e := ir.Event{Type: ir.EventStart}
		if ev.Response != nil {
			e.ID, e.Model = ev.Response.ID, ev.Response.Model
		}
		out = append(out, e)
	}
	switch ev.Type {
	case "response.output_item.added":
		it := ev.Item
		if it == nil {
			break
		}
		st := &itemState{irIdx: -1, typ: it.Type, callID: it.CallID, textIdx: map[int]int{}}
		d.items[it.ID] = st
		d.byOut[ev.OutputIndex] = st
		switch it.Type {
		case "reasoning":
			idx, e := d.block(ir.Part{Type: ir.PartThinking, SigOrigin: self})
			st.irIdx = idx
			out = append(out, e)
		case "function_call", "custom_tool_call":
			d.sawTool = true
			st.custom = it.Type == "custom_tool_call"
			idx, e := d.block(ir.Part{Type: ir.PartToolCall, ToolCallID: it.CallID, ToolName: it.Name})
			st.irIdx = idx
			out = append(out, e)
			if st.custom {
				st.args.WriteString(`{"input":"`)
				out = append(out, ir.Event{Type: ir.EventDelta, Index: idx, Delta: ir.DeltaToolArgs, Text: `{"input":"`})
			}
		}
	case "response.output_text.delta", "response.refusal.delta":
		st := d.item(&ev)
		if st == nil || ev.Delta == "" {
			break
		}
		idx, ok := st.textIdx[ev.ContentIndex]
		if !ok {
			var e ir.Event
			idx, e = d.block(ir.Part{Type: ir.PartText})
			st.textIdx[ev.ContentIndex] = idx
			st.hadText = true
			out = append(out, e)
		}
		out = append(out, ir.Event{Type: ir.EventDelta, Index: idx, Delta: ir.DeltaText, Text: ev.Delta})
	case "response.content_part.done":
		st := d.item(&ev)
		if st == nil {
			break
		}
		if idx, ok := st.textIdx[ev.ContentIndex]; ok {
			out = append(out, ir.Event{Type: ir.EventBlockStop, Index: idx})
			delete(st.textIdx, ev.ContentIndex)
		}
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		st := d.item(&ev)
		if st == nil || st.irIdx < 0 || ev.Delta == "" {
			break
		}
		text := ev.Delta
		if ev.SummaryIndex > st.summaryIx {
			st.summaryIx = ev.SummaryIndex
			text = "\n\n" + text
		}
		out = append(out, ir.Event{Type: ir.EventDelta, Index: st.irIdx, Delta: ir.DeltaThinking, Text: text})
	case "response.function_call_arguments.delta", "response.custom_tool_call_input.delta":
		st := d.item(&ev)
		if st == nil || st.irIdx < 0 || ev.Delta == "" {
			break
		}
		text := ev.Delta
		if st.custom {
			text = jsonFragment(text)
		}
		st.args.WriteString(text)
		out = append(out, ir.Event{Type: ir.EventDelta, Index: st.irIdx, Delta: ir.DeltaToolArgs, Text: text})
	case "response.output_item.done":
		it := ev.Item
		st := d.item(&ev)
		if it != nil && st == nil {
			st = d.items[it.ID]
		}
		if st == nil || it == nil {
			break
		}
		out = append(out, d.finishItem(st, it)...)
	case "response.completed", "response.incomplete", "response.done":
		d.done = true
		stop := ir.StopEndTurn
		if r := ev.Response; r != nil {
			reason := ""
			if r.IncompleteDetails != nil {
				reason = r.IncompleteDetails.Reason
			}
			status := r.Status
			if ev.Type == "response.incomplete" {
				status = "incomplete"
			}
			stop = stopFrom(status, reason, d.sawTool)
			if r.Usage != nil {
				u := r.Usage.ir()
				out = append(out, ir.Event{Type: ir.EventUsage, Usage: &u})
			}
		} else if d.sawTool {
			stop = ir.StopToolUse
		}
		out = append(out, ir.Event{Type: ir.EventStop, StopReason: stop})
	case "response.failed":
		d.done = true
		msg := "upstream response failed"
		if ev.Response != nil && ev.Response.Error != nil {
			msg = ev.Response.Error.Message
		}
		out = append(out, ir.Event{Type: ir.EventError, Err: &ir.APIError{Status: 502, Type: "api_error", Message: msg}})
	case "error":
		d.done = true
		msg := ev.Message
		if msg == "" {
			msg = "upstream stream error"
		}
		out = append(out, ir.Event{Type: ir.EventError, Err: &ir.APIError{Status: 500, Type: "api_error", Message: msg}})
	}
	return out, nil
}

func (d *decoder) finishItem(st *itemState, it *wireItem) []ir.Event {
	var out []ir.Event
	switch it.Type {
	case "reasoning":
		if st.irIdx < 0 {
			return nil
		}
		if sig := packReasoning(it.ID, it.EncryptedContent); sig != "" {
			out = append(out, ir.Event{Type: ir.EventDelta, Index: st.irIdx, Delta: ir.DeltaSignature, Text: sig})
		}
		out = append(out, ir.Event{Type: ir.EventBlockStop, Index: st.irIdx})
	case "function_call", "custom_tool_call":
		if st.irIdx < 0 {
			return nil
		}
		full := it.Arguments
		if st.custom {
			full = `{"input":"` + jsonFragment(it.Input)
		}
		// Emit only what was not streamed already.
		if streamed := st.args.String(); strings.HasPrefix(full, streamed) && len(full) > len(streamed) {
			out = append(out, ir.Event{Type: ir.EventDelta, Index: st.irIdx, Delta: ir.DeltaToolArgs, Text: full[len(streamed):]})
		}
		if st.custom {
			out = append(out, ir.Event{Type: ir.EventDelta, Index: st.irIdx, Delta: ir.DeltaToolArgs, Text: `"}`})
		}
		out = append(out, ir.Event{Type: ir.EventBlockStop, Index: st.irIdx})
	case "message":
		// Recover text that only appeared in the final item.
		if !st.hadText && len(it.Content) > 0 {
			var cs []wireContent
			_ = json.Unmarshal(it.Content, &cs)
			for _, c := range cs {
				t := c.Text
				if c.Type == "refusal" {
					t = c.Refusal
				}
				if t == "" {
					continue
				}
				idx, e := d.block(ir.Part{Type: ir.PartText})
				out = append(out, e, ir.Event{Type: ir.EventDelta, Index: idx, Delta: ir.DeltaText, Text: t},
					ir.Event{Type: ir.EventBlockStop, Index: idx})
			}
			st.hadText = true
		}
		for ci, idx := range st.textIdx {
			out = append(out, ir.Event{Type: ir.EventBlockStop, Index: idx})
			delete(st.textIdx, ci)
		}
	}
	return out
}

func (d *decoder) Finish() ([]ir.Event, error) {
	if d.done {
		return nil, nil
	}
	return nil, convert.ErrIncompleteStream
}

// ---- encoder: IR -> Responses SSE ----

type outItem struct {
	outIndex int
	id       string
	part     ir.Part
	text     strings.Builder
	sig      strings.Builder
	item     obj // final item
}

type encoder struct {
	req      *ir.Request
	id       string
	model    string
	seq      int
	started  bool
	finished bool
	items    map[int]*outItem // ir index -> item
	output   []obj            // completed items in order
	nextOut  int
	usage    ir.Usage
	sawTool  bool
}

func (Codec) NewStreamEncoder(req *ir.Request) convert.StreamEncoder {
	e := &encoder{req: req, items: map[int]*outItem{}}
	if req != nil {
		e.model = req.Model
	}
	return e
}

func (e *encoder) ev(typ string, payload obj) sse.Event {
	payload["type"] = typ
	payload["sequence_number"] = e.seq
	e.seq++
	b, _ := json.Marshal(payload)
	return sse.Event{Event: typ, Data: string(b)}
}

func (e *encoder) envelope(status string) obj {
	out := e.output
	if out == nil {
		out = []obj{}
	}
	return responseEnvelope(e.id, e.model, status, out)
}

func (e *encoder) start(id, model string) []sse.Event {
	if e.started {
		return nil
	}
	e.started = true
	e.id = respID(id)
	if e.model == "" {
		e.model = model
	}
	return []sse.Event{
		e.ev("response.created", obj{"response": e.envelope("in_progress")}),
		e.ev("response.in_progress", obj{"response": e.envelope("in_progress")}),
	}
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
	case ir.EventStart:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
	case ir.EventBlockStart:
		out = append(out, e.openItem(ev.Index, *ev.Block)...)
	case ir.EventDelta:
		it := e.items[ev.Index]
		if it == nil {
			break
		}
		switch ev.Delta {
		case ir.DeltaText:
			it.text.WriteString(ev.Text)
			out = append(out, e.ev("response.output_text.delta", obj{"item_id": it.id, "output_index": it.outIndex,
				"content_index": 0, "delta": ev.Text, "logprobs": []any{}}))
		case ir.DeltaThinking:
			it.text.WriteString(ev.Text)
			out = append(out, e.ev("response.reasoning_summary_text.delta", obj{"item_id": it.id, "output_index": it.outIndex,
				"summary_index": 0, "delta": ev.Text}))
		case ir.DeltaSignature:
			it.sig.WriteString(ev.Text)
		case ir.DeltaToolArgs:
			it.text.WriteString(ev.Text)
			if !it.isCustom(e.req) {
				out = append(out, e.ev("response.function_call_arguments.delta", obj{"item_id": it.id, "output_index": it.outIndex, "delta": ev.Text}))
			}
		}
	case ir.EventBlockStop:
		out = append(out, e.closeItem(ev.Index)...)
	case ir.EventUsage:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
	case ir.EventStop:
		if ev.Usage != nil {
			e.usage.Merge(*ev.Usage)
		}
		out = append(out, e.finish(ev.StopReason)...)
	case ir.EventError:
		e.finished = true
		msg := "stream error"
		if ev.Err != nil {
			msg = ev.Err.Message
		}
		if !e.started {
			e.started = true
			e.id = respID("")
		}
		env := e.envelope("failed")
		env["error"] = obj{"code": "server_error", "message": msg}
		out = append(out, e.ev("response.failed", obj{"response": env}))
	}
	return out
}

func (it *outItem) isCustom(req *ir.Request) bool {
	t := req.ToolByName(it.part.ToolName)
	return it.part.Type == ir.PartToolCall && t != nil && t.Custom
}

func (e *encoder) openItem(idx int, p ir.Part) []sse.Event {
	it := &outItem{outIndex: e.nextOut, part: p}
	var added obj
	switch p.Type {
	case ir.PartText:
		it.id = ir.NewID("msg_")
		added = obj{"id": it.id, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}
	case ir.PartThinking, ir.PartRedactedThinking:
		if p.SigOrigin == self {
			it.id, _ = unpackReasoning(p.Signature)
		}
		if it.id == "" {
			it.id = ir.NewID("rs_")
		}
		if p.Type == ir.PartRedactedThinking {
			it.sig.WriteString(p.Signature)
		}
		added = obj{"id": it.id, "type": "reasoning", "summary": []any{}}
	case ir.PartToolCall:
		e.sawTool = true
		it.id = ir.NewID("fc_")
		item := callItem(p, e.req, "in_progress")
		item["id"] = it.id
		if item["type"] == "function_call" {
			item["arguments"] = ""
		} else {
			item["input"] = ""
		}
		added = item
	default:
		return nil
	}
	e.nextOut++
	e.items[idx] = it
	out := []sse.Event{e.ev("response.output_item.added", obj{"output_index": it.outIndex, "item": added})}
	switch p.Type {
	case ir.PartText:
		out = append(out, e.ev("response.content_part.added", obj{"item_id": it.id, "output_index": it.outIndex, "content_index": 0,
			"part": obj{"type": "output_text", "text": "", "annotations": []any{}}}))
	case ir.PartThinking:
		out = append(out, e.ev("response.reasoning_summary_part.added", obj{"item_id": it.id, "output_index": it.outIndex, "summary_index": 0,
			"part": obj{"type": "summary_text", "text": ""}}))
	}
	return out
}

func (e *encoder) closeItem(idx int) []sse.Event {
	it := e.items[idx]
	if it == nil {
		return nil
	}
	delete(e.items, idx)
	var out []sse.Event
	var item obj
	switch it.part.Type {
	case ir.PartText:
		text := it.text.String()
		part := obj{"type": "output_text", "text": text, "annotations": []any{}}
		out = append(out,
			e.ev("response.output_text.done", obj{"item_id": it.id, "output_index": it.outIndex, "content_index": 0, "text": text, "logprobs": []any{}}),
			e.ev("response.content_part.done", obj{"item_id": it.id, "output_index": it.outIndex, "content_index": 0, "part": part}))
		item = messageItem(it.id, text, "completed")
	case ir.PartThinking, ir.PartRedactedThinking:
		text := it.text.String()
		if it.part.Type == ir.PartThinking {
			out = append(out,
				e.ev("response.reasoning_summary_text.done", obj{"item_id": it.id, "output_index": it.outIndex, "summary_index": 0, "text": text}),
				e.ev("response.reasoning_summary_part.done", obj{"item_id": it.id, "output_index": it.outIndex, "summary_index": 0,
					"part": obj{"type": "summary_text", "text": text}}))
		}
		p := it.part
		p.Text = text
		p.Signature = it.sig.String()
		item = reasoningItem(p)
		if p.SigOrigin != self {
			item["id"] = it.id // keep the id announced in output_item.added
		}
	case ir.PartToolCall:
		p := it.part
		p.Arguments = it.text.String()
		item = callItem(p, e.req, "completed")
		item["id"] = it.id
		if item["type"] == "function_call" {
			out = append(out, e.ev("response.function_call_arguments.done", obj{"item_id": it.id, "output_index": it.outIndex, "arguments": item["arguments"]}))
		} else {
			out = append(out, e.ev("response.custom_tool_call_input.done", obj{"item_id": it.id, "output_index": it.outIndex, "input": item["input"]}))
		}
	}
	e.output = append(e.output, item)
	return append(out, e.ev("response.output_item.done", obj{"output_index": it.outIndex, "item": item}))
}

func (e *encoder) finish(stop ir.StopReason) []sse.Event {
	out := e.start("", "")
	// close remaining items in output order
	for len(e.items) > 0 {
		minIdx, minOut := -1, 1<<30
		for idx, it := range e.items {
			if it.outIndex < minOut {
				minIdx, minOut = idx, it.outIndex
			}
		}
		out = append(out, e.closeItem(minIdx)...)
	}
	status, incomplete := statusOf(stop)
	env := e.envelope(status)
	env["incomplete_details"] = incomplete
	env["usage"] = usageObj(e.usage)
	name := "response.completed"
	if status == "incomplete" {
		name = "response.incomplete"
	}
	out = append(out, e.ev(name, obj{"response": env}))
	e.finished = true
	return out
}

func (e *encoder) Finalize() []sse.Event {
	if e.finished {
		return nil
	}
	return e.finish(ir.StopEndTurn)
}
