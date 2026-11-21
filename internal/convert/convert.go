// Package convert defines the codec contract every wire format implements and
// a registry to look codecs up by format. Codecs live in sub-packages and
// register themselves in init(); import convert/all to get every codec.
package convert

import (
	"errors"
	"fmt"
	"sync"

	"llm-one-api/internal/convert/ir"
	"llm-one-api/internal/convert/sse"
)

// ErrIncompleteStream is returned by StreamDecoder.Finish when the upstream
// stream ended without its terminal marker.
var ErrIncompleteStream = errors.New("upstream stream ended prematurely")

// StreamDecoder turns upstream SSE events into IR events.
type StreamDecoder interface {
	Feed(ev sse.Event) ([]ir.Event, error)
	// Finish is called at EOF. It flushes pending events and returns
	// ErrIncompleteStream if no terminal marker was seen.
	Finish() ([]ir.Event, error)
}

// StreamEncoder turns IR events into client SSE events.
type StreamEncoder interface {
	Encode(ev ir.Event) []sse.Event
	// Finalize closes anything still open. Safe to call more than once.
	Finalize() []sse.Event
}

// Codec converts one wire format to and from the IR.
type Codec interface {
	Format() ir.Format

	DecodeRequest(body []byte) (*ir.Request, error)
	EncodeRequest(req *ir.Request) ([]byte, error)

	DecodeResponse(body []byte) (*ir.Response, error)
	// EncodeResponse renders resp for a client; req is the decoded client
	// request (may be nil) and is used for context such as tool kinds.
	EncodeResponse(resp *ir.Response, req *ir.Request) ([]byte, error)

	NewStreamDecoder() StreamDecoder
	NewStreamEncoder(req *ir.Request) StreamEncoder

	DecodeError(status int, body []byte) *ir.APIError
	EncodeError(e *ir.APIError) []byte
}

var (
	mu       sync.RWMutex
	registry = map[ir.Format]Codec{}
)

// Register adds a codec. It panics on duplicates.
func Register(c Codec) {
	mu.Lock()
	defer mu.Unlock()
	if _, dup := registry[c.Format()]; dup {
		panic(fmt.Sprintf("convert: codec %q registered twice", c.Format()))
	}
	registry[c.Format()] = c
}

// Get returns the codec for f.
func Get(f ir.Format) (Codec, error) {
	mu.RLock()
	defer mu.RUnlock()
	c, ok := registry[f]
	if !ok {
		return nil, fmt.Errorf("convert: no codec for format %q", f)
	}
	return c, nil
}

// MustGet is Get that panics.
func MustGet(f ir.Format) Codec {
	c, err := Get(f)
	if err != nil {
		panic(err)
	}
	return c
}

// Request converts a request body between formats via the IR. The decoded
// IR request (with diagnostics) is returned alongside the encoded body.
func Request(from, to ir.Format, body []byte, policy ir.LossPolicy) ([]byte, *ir.Request, error) {
	src, err := Get(from)
	if err != nil {
		return nil, nil, err
	}
	dst, err := Get(to)
	if err != nil {
		return nil, nil, err
	}
	req, err := src.DecodeRequest(body)
	if err != nil {
		return nil, nil, err
	}
	out, err := dst.EncodeRequest(req)
	if err != nil {
		return nil, req, err
	}
	if err := policy.Check(req.Diags); err != nil {
		return nil, req, err
	}
	return out, req, nil
}

// Response converts a non-streaming response body between formats.
func Response(from, to ir.Format, body []byte, clientReq *ir.Request) ([]byte, error) {
	src, err := Get(from)
	if err != nil {
		return nil, err
	}
	dst, err := Get(to)
	if err != nil {
		return nil, err
	}
	resp, err := src.DecodeResponse(body)
	if err != nil {
		return nil, err
	}
	return dst.EncodeResponse(resp, clientReq)
}

// Pipe converts a full sequence of upstream SSE events to client SSE events.
// Intended for tests and buffered use; the relay streams incrementally.
func Pipe(from, to ir.Format, events []sse.Event, clientReq *ir.Request) ([]sse.Event, error) {
	src, err := Get(from)
	if err != nil {
		return nil, err
	}
	dst, err := Get(to)
	if err != nil {
		return nil, err
	}
	dec := src.NewStreamDecoder()
	enc := dst.NewStreamEncoder(clientReq)
	var out []sse.Event
	for _, ev := range events {
		irs, err := dec.Feed(ev)
		for _, e := range irs {
			out = append(out, enc.Encode(e)...)
		}
		if err != nil {
			return out, err
		}
	}
	irs, ferr := dec.Finish()
	for _, e := range irs {
		out = append(out, enc.Encode(e)...)
	}
	if ferr != nil {
		out = append(out, enc.Encode(ir.Event{Type: ir.EventError, Err: &ir.APIError{Status: 502, Type: "api_error", Message: ferr.Error()}})...)
	}
	out = append(out, enc.Finalize()...)
	return out, nil
}
