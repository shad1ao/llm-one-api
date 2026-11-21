package ir

// EventType enumerates neutral streaming events.
//
// Invariants decoders guarantee:
//   - EventStart comes first (exactly once).
//   - A block's EventBlockStart precedes its deltas and its EventBlockStop.
//   - Tool-call blocks are only started once both id and name are known.
//   - Final usage is delivered via EventUsage before EventStop.
//   - EventStop is the last event of a successful stream.
type EventType int

const (
	EventStart EventType = iota
	EventBlockStart
	EventDelta
	EventBlockStop
	EventUsage
	EventStop
	EventError
)

type DeltaType int

const (
	DeltaText DeltaType = iota
	DeltaThinking
	DeltaSignature
	DeltaToolArgs
)

type Event struct {
	Type EventType

	// EventStart
	ID    string
	Model string

	// EventBlockStart / EventDelta / EventBlockStop
	Index int
	// EventBlockStart: Block.Type plus ToolCallID/ToolName for tool calls,
	// Signature for redacted thinking. No incremental content.
	Block *Part

	// EventDelta
	Delta DeltaType
	Text  string

	// EventUsage / EventStop
	Usage *Usage

	// EventStop
	StopReason StopReason
	StopSeq    string

	// EventError
	Err *APIError
}

// Collector accumulates a stream of IR events into a Response. Useful for
// turning a streaming upstream into a non-streaming client reply.
type Collector struct {
	resp    Response
	byIndex map[int]int // ir index -> position in resp.Parts
	done    bool
	err     *APIError
}

func NewCollector() *Collector { return &Collector{byIndex: map[int]int{}} }

func (c *Collector) Add(ev Event) {
	switch ev.Type {
	case EventStart:
		if ev.ID != "" {
			c.resp.ID = ev.ID
		}
		if ev.Model != "" {
			c.resp.Model = ev.Model
		}
		if ev.Usage != nil {
			c.resp.Usage.Merge(*ev.Usage)
		}
	case EventBlockStart:
		p := *ev.Block
		c.byIndex[ev.Index] = len(c.resp.Parts)
		c.resp.Parts = append(c.resp.Parts, p)
	case EventDelta:
		pos, ok := c.byIndex[ev.Index]
		if !ok {
			return
		}
		p := &c.resp.Parts[pos]
		switch ev.Delta {
		case DeltaText, DeltaThinking:
			p.Text += ev.Text
		case DeltaSignature:
			p.Signature += ev.Text
		case DeltaToolArgs:
			p.Arguments += ev.Text
		}
	case EventUsage:
		if ev.Usage != nil {
			c.resp.Usage.Merge(*ev.Usage)
		}
	case EventStop:
		c.done = true
		c.resp.StopReason = ev.StopReason
		c.resp.StopSeq = ev.StopSeq
		if ev.Usage != nil {
			c.resp.Usage.Merge(*ev.Usage)
		}
	case EventError:
		c.err = ev.Err
	}
}

// Result returns the collected response, or the stream error.
func (c *Collector) Result() (*Response, *APIError) {
	if c.err != nil {
		return nil, c.err
	}
	r := c.resp
	for i := range r.Parts {
		if r.Parts[i].Type == PartToolCall && r.Parts[i].Arguments == "" {
			r.Parts[i].Arguments = "{}"
		}
	}
	if r.StopReason == "" {
		if r.HasToolCalls() {
			r.StopReason = StopToolUse
		} else {
			r.StopReason = StopEndTurn
		}
	}
	return &r, nil
}

// Done reports whether a terminal EventStop was seen.
func (c *Collector) Done() bool { return c.done }
