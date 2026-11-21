package relay

import (
	"io"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
	"llm-one-api/internal/convert/sse"
)

// Stream converts an upstream SSE stream into the client format and writes
// each event as soon as it is available. flush, when non-nil, is called after
// every write that produced bytes. A truncated upstream stream is finished
// with an error event rather than returned as a Go error.
func Stream(from, to ir.Format, src io.Reader, dst io.Writer, flush func(), clientReq *ir.Request) error {
	dec := convert.MustGet(from).NewStreamDecoder()
	enc := convert.MustGet(to).NewStreamEncoder(clientReq)
	rd := sse.NewReader(src)

	writeIR := func(evs []ir.Event) error {
		var out []sse.Event
		for _, ev := range evs {
			out = append(out, enc.Encode(ev)...)
		}
		if err := writeSSE(dst, out); err != nil {
			return err
		}
		if flush != nil && len(out) > 0 {
			flush()
		}
		return nil
	}
	fail := func(err error) error {
		_ = writeIR([]ir.Event{{Type: ir.EventError, Err: &ir.APIError{Status: 502, Type: "api_error", Message: err.Error()}}})
		if err := writeSSE(dst, enc.Finalize()); err != nil {
			return err
		}
		if flush != nil {
			flush()
		}
		return nil
	}

	for {
		ev, err := rd.Next()
		if err != nil {
			break
		}
		irs, ferr := dec.Feed(ev)
		if werr := writeIR(irs); werr != nil {
			return werr
		}
		if ferr != nil {
			return fail(ferr)
		}
	}
	irs, ferr := dec.Finish()
	if werr := writeIR(irs); werr != nil {
		return werr
	}
	if ferr != nil {
		return fail(ferr)
	}
	if err := writeSSE(dst, enc.Finalize()); err != nil {
		return err
	}
	if flush != nil {
		flush()
	}
	return nil
}

// Collect folds an upstream SSE stream into one response.
func Collect(from ir.Format, src io.Reader) (*ir.Response, *ir.APIError, error) {
	dec := convert.MustGet(from).NewStreamDecoder()
	c := ir.NewCollector()
	rd := sse.NewReader(src)
	feed := func(evs []ir.Event, ferr error) (*ir.Response, *ir.APIError, error) {
		for _, ev := range evs {
			c.Add(ev)
		}
		if ferr != nil {
			return nil, &ir.APIError{Status: 502, Type: "api_error", Message: ferr.Error()}, nil
		}
		return nil, nil, nil
	}
	for {
		ev, err := rd.Next()
		if err != nil {
			break
		}
		irs, ferr := dec.Feed(ev)
		if resp, apiErr, ferr2 := feed(irs, ferr); resp != nil || apiErr != nil || ferr2 != nil {
			return resp, apiErr, ferr2
		}
	}
	irs, ferr := dec.Finish()
	if _, apiErr, err := feed(irs, ferr); apiErr != nil || err != nil {
		return nil, apiErr, err
	}
	resp, apiErr := c.Result()
	return resp, apiErr, nil
}

func writeSSE(dst io.Writer, evs []sse.Event) error {
	for _, ev := range evs {
		if _, err := dst.Write(ev.Bytes()); err != nil {
			return err
		}
	}
	return nil
}

// eventsFromResponse renders a non-streaming response as IR events so a
// client that asked to stream still receives SSE when the upstream did not.
func eventsFromResponse(resp *ir.Response) []ir.Event {
	evs := []ir.Event{{Type: ir.EventStart, ID: resp.ID, Model: resp.Model}}
	idx := 0
	for _, p := range resp.Parts {
		start := p
		text, args := p.Text, p.Arguments
		start.Text = ""
		if p.Type != ir.PartToolCall {
			start.Arguments = ""
		}
		if p.Type == ir.PartThinking {
			start.Signature = ""
		}
		switch p.Type {
		case ir.PartText, ir.PartThinking, ir.PartRedactedThinking, ir.PartToolCall:
		default:
			continue
		}
		evs = append(evs, ir.Event{Type: ir.EventBlockStart, Index: idx, Block: &start})
		switch p.Type {
		case ir.PartText:
			if text != "" {
				evs = append(evs, ir.Event{Type: ir.EventDelta, Index: idx, Delta: ir.DeltaText, Text: text})
			}
		case ir.PartThinking:
			if text != "" {
				evs = append(evs, ir.Event{Type: ir.EventDelta, Index: idx, Delta: ir.DeltaThinking, Text: text})
			}
			if p.Signature != "" {
				evs = append(evs, ir.Event{Type: ir.EventDelta, Index: idx, Delta: ir.DeltaSignature, Text: p.Signature})
			}
		case ir.PartToolCall:
			if args == "" {
				args = "{}"
			}
			evs = append(evs, ir.Event{Type: ir.EventDelta, Index: idx, Delta: ir.DeltaToolArgs, Text: args})
		}
		evs = append(evs, ir.Event{Type: ir.EventBlockStop, Index: idx})
		idx++
	}
	u := resp.Usage
	evs = append(evs,
		ir.Event{Type: ir.EventUsage, Usage: &u},
		ir.Event{Type: ir.EventStop, StopReason: resp.StopReason, StopSeq: resp.StopSeq, Usage: &u},
	)
	return evs
}

func writeEvents(to ir.Format, evs []ir.Event, dst io.Writer, flush func(), clientReq *ir.Request) error {
	enc := convert.MustGet(to).NewStreamEncoder(clientReq)
	var out []sse.Event
	for _, ev := range evs {
		out = append(out, enc.Encode(ev)...)
	}
	out = append(out, enc.Finalize()...)
	if err := writeSSE(dst, out); err != nil {
		return err
	}
	if flush != nil && len(out) > 0 {
		flush()
	}
	return nil
}
