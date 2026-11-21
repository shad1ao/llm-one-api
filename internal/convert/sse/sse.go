// Package sse implements a minimal Server-Sent Events reader and writer.
package sse

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

// Event is one SSE message. Event is empty for unnamed events.
type Event struct {
	Event string
	Data  string
}

// Bytes serializes the event in wire form, terminated by a blank line.
func (e Event) Bytes() []byte {
	var b bytes.Buffer
	if e.Event != "" {
		b.WriteString("event: ")
		b.WriteString(e.Event)
		b.WriteByte('\n')
	}
	for line := range strings.SplitSeq(e.Data, "\n") {
		b.WriteString("data: ")
		b.WriteString(line)
		b.WriteByte('\n')
	}
	b.WriteByte('\n')
	return b.Bytes()
}

// Reader parses SSE events from a stream. Lines of any length are supported.
type Reader struct {
	r *bufio.Reader
}

func NewReader(r io.Reader) *Reader { return &Reader{r: bufio.NewReaderSize(r, 64*1024)} }

// Next returns the next event. It returns io.EOF when the stream ends with no
// pending event.
func (rd *Reader) Next() (Event, error) {
	var ev Event
	var data []string
	hasData := false
	for {
		line, err := rd.r.ReadString('\n')
		if err != nil && line == "" {
			if hasData || ev.Event != "" {
				ev.Data = strings.Join(data, "\n")
				return ev, nil
			}
			return Event{}, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if hasData || ev.Event != "" {
				ev.Data = strings.Join(data, "\n")
				return ev, nil
			}
			if err != nil {
				return Event{}, err
			}
			continue
		}
		if line[0] == ':' {
			continue // comment / keepalive
		}
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			ev.Event = value
		case "data":
			data = append(data, value)
			hasData = true
		}
		if err != nil {
			ev.Data = strings.Join(data, "\n")
			return ev, nil
		}
	}
}
