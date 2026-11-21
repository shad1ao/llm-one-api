package claude

import (
	"encoding/json"
	"fmt"
	"strings"

	"llm-one-api/internal/convert/ir"
)

type wireUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreation            *struct {
		Ephemeral5m int `json:"ephemeral_5m_input_tokens"`
		Ephemeral1h int `json:"ephemeral_1h_input_tokens"`
	} `json:"cache_creation"`
}

func (u wireUsage) ir() ir.Usage {
	out := ir.Usage{
		Input:      u.InputTokens,
		Output:     u.OutputTokens,
		CacheRead:  u.CacheReadInputTokens,
		CacheWrite: u.CacheCreationInputTokens,
	}
	if u.CacheCreation != nil {
		out.CacheWrite1h = u.CacheCreation.Ephemeral1h
		if out.CacheWrite == 0 {
			out.CacheWrite = u.CacheCreation.Ephemeral5m + u.CacheCreation.Ephemeral1h
		}
	}
	return out
}

func usageObj(u ir.Usage) obj {
	o := obj{
		"input_tokens":                u.Input,
		"output_tokens":               u.Output,
		"cache_creation_input_tokens": u.CacheWrite,
		"cache_read_input_tokens":     u.CacheRead,
	}
	if u.CacheWrite > 0 {
		o["cache_creation"] = obj{
			"ephemeral_5m_input_tokens": u.CacheWrite - u.CacheWrite1h,
			"ephemeral_1h_input_tokens": u.CacheWrite1h,
		}
	}
	return o
}

type wireResponse struct {
	ID           string      `json:"id"`
	Type         string      `json:"type"`
	Model        string      `json:"model"`
	Content      []wireBlock `json:"content"`
	StopReason   string      `json:"stop_reason"`
	StopSequence string      `json:"stop_sequence"`
	Usage        wireUsage   `json:"usage"`
}

func (Codec) DecodeResponse(body []byte) (*ir.Response, error) {
	var w wireResponse
	if err := json.Unmarshal(body, &w); err != nil {
		return nil, fmt.Errorf("invalid claude response: %w", err)
	}
	resp := &ir.Response{ID: w.ID, Model: w.Model, StopReason: ir.StopFromClaude(w.StopReason),
		StopSeq: w.StopSequence, Usage: w.Usage.ir()}
	for _, b := range w.Content {
		switch b.Type {
		case "text":
			resp.Parts = append(resp.Parts, ir.Part{Type: ir.PartText, Text: b.Text})
		case "thinking":
			resp.Parts = append(resp.Parts, ir.Part{Type: ir.PartThinking, Text: b.Thinking, Signature: b.Signature, SigOrigin: self})
		case "redacted_thinking":
			resp.Parts = append(resp.Parts, ir.Part{Type: ir.PartRedactedThinking, Signature: b.Data, SigOrigin: self})
		case "tool_use":
			resp.Parts = append(resp.Parts, ir.Part{Type: ir.PartToolCall, ToolCallID: b.ID, ToolName: b.Name, Arguments: ir.RawString(b.Input)})
		}
	}
	return resp, nil
}

func (Codec) EncodeResponse(resp *ir.Response, _ *ir.Request) ([]byte, error) {
	content := make([]obj, 0, len(resp.Parts))
	for _, p := range resp.Parts {
		if b := responseBlock(p); b != nil {
			content = append(content, b)
		}
	}
	id := resp.ID
	if !strings.HasPrefix(id, "msg_") {
		id = ir.NewID("msg_")
	}
	stop := resp.StopReason
	if stop == ir.StopEndTurn && resp.HasToolCalls() {
		stop = ir.StopToolUse
	}
	out := obj{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         resp.Model,
		"content":       content,
		"stop_reason":   ir.StopToClaude(stop),
		"stop_sequence": nil,
		"usage":         usageObj(resp.Usage),
	}
	if resp.StopSeq != "" {
		out["stop_sequence"] = resp.StopSeq
	}
	return json.Marshal(out)
}

// responseBlock renders a part for a Claude client.
func responseBlock(p ir.Part) obj {
	switch p.Type {
	case ir.PartText:
		return obj{"type": "text", "text": p.Text}
	case ir.PartThinking:
		return obj{"type": "thinking", "thinking": p.Text, "signature": ir.SigFor(self, p.SigOrigin, p.Signature)}
	case ir.PartRedactedThinking:
		if p.SigOrigin == self {
			return obj{"type": "redacted_thinking", "data": p.Signature}
		}
		// Opaque foreign reasoning: keep it replayable via an empty thinking block.
		return obj{"type": "thinking", "thinking": "", "signature": ir.SigFor(self, p.SigOrigin, p.Signature)}
	case ir.PartToolCall:
		args, _ := ir.NormalizeArgs(p.Arguments)
		if !json.Valid([]byte(args)) {
			args = "{}"
		}
		return obj{"type": "tool_use", "id": clientToolID(p), "name": p.ToolName, "input": json.RawMessage(args)}
	}
	return nil
}

func clientToolID(p ir.Part) string {
	id := p.ToolCallID
	if id == "" {
		id = ir.NewID("toolu_")
	}
	return ir.PackCallID(self, toolID(id), p.SigOrigin, p.Signature)
}

// ---- errors ----

func (Codec) DecodeError(status int, body []byte) *ir.APIError {
	var w struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	e := &ir.APIError{Status: status, Type: ir.ErrorTypeForStatus(status)}
	if json.Unmarshal(body, &w) == nil && w.Error.Message != "" {
		e.Message = w.Error.Message
		if w.Error.Type != "" {
			e.Type = w.Error.Type
		}
		return e
	}
	e.Message = strings.TrimSpace(string(body))
	if e.Message == "" {
		e.Message = fmt.Sprintf("upstream returned status %d", status)
	}
	return e
}

func (Codec) EncodeError(e *ir.APIError) []byte {
	b, _ := json.Marshal(obj{"type": "error", "error": obj{"type": claudeErrType(e), "message": e.Message}})
	return b
}

func claudeErrType(e *ir.APIError) string {
	switch e.Type {
	case "invalid_request_error", "authentication_error", "permission_error", "not_found_error",
		"request_too_large", "rate_limit_error", "api_error", "overloaded_error":
		return e.Type
	}
	return ir.ErrorTypeForStatus(e.Status)
}
