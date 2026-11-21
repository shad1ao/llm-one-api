package ir

// RepairToolPairing makes a conversation acceptable to strict providers:
//   - consecutive same-role messages are merged;
//   - every tool call in an assistant message gets a tool result at the start
//     of the following user message (a placeholder error is synthesized if the
//     result is missing);
//   - tool results that do not answer a call in the preceding assistant
//     message are turned into plain text.
func RepairToolPairing(msgs []Message, d *Diags) []Message {
	msgs = MergeAdjacent(msgs)
	out := make([]Message, 0, len(msgs)+1)
	for _, m := range msgs {
		if m.Role == RoleUser {
			var expected []Part
			if n := len(out); n > 0 && out[n-1].Role == RoleAssistant {
				expected = toolCalls(out[n-1].Parts)
			}
			m.Parts = fixResults(m.Parts, expected, d)
		}
		// Trailing assistant tool calls (no following user turn) are left as is.
		out = append(out, m)
	}
	return out
}

func toolCalls(parts []Part) []Part {
	var calls []Part
	for _, p := range parts {
		if p.Type == PartToolCall {
			calls = append(calls, p)
		}
	}
	return calls
}

func fixResults(parts []Part, calls []Part, d *Diags) []Part {
	want := map[string]bool{}
	for _, c := range calls {
		want[c.ToolCallID] = true
	}
	results := map[string]Part{}
	var rest []Part
	for _, p := range parts {
		if p.Type != PartToolResult {
			rest = append(rest, p)
			continue
		}
		if want[p.ToolCallID] {
			if _, dup := results[p.ToolCallID]; !dup {
				results[p.ToolCallID] = p
				continue
			}
		}
		// Orphan result: degrade to text.
		if d != nil {
			d.Add(SevInfo, "orphan_tool_result", "tool result "+p.ToolCallID+" has no matching call; sent as text")
		}
		txt := "[tool result " + p.ToolCallID + "]\n" + TextOf(p.Content)
		rest = append(rest, Part{Type: PartText, Text: txt})
	}
	if len(calls) == 0 {
		return rest
	}
	fixed := make([]Part, 0, len(parts)+len(calls))
	for _, c := range calls {
		r, ok := results[c.ToolCallID]
		if !ok {
			if d != nil {
				d.Add(SevInfo, "missing_tool_result", "tool call "+c.ToolCallID+" has no result; placeholder inserted")
			}
			r = Part{Type: PartToolResult, ToolCallID: c.ToolCallID, ToolName: c.ToolName, IsError: true,
				Content: []Part{{Type: PartText, Text: "tool result missing"}}}
		}
		if r.ToolName == "" {
			r.ToolName = c.ToolName
		}
		fixed = append(fixed, r)
	}
	return append(fixed, rest...)
}
