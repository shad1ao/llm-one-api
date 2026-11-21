package openaichat

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"llm-one-api/internal/convert/ir"
)

type wireUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails *struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
	PromptCacheHitTokens int `json:"prompt_cache_hit_tokens"` // DeepSeek
}

func (u wireUsage) ir() ir.Usage {
	cached, write, reasoning := u.PromptCacheHitTokens, 0, 0
	if d := u.PromptTokensDetails; d != nil {
		if d.CachedTokens > 0 {
			cached = d.CachedTokens
		}
		write = d.CacheWriteTokens
	}
	if d := u.CompletionTokensDetails; d != nil {
		reasoning = d.ReasoningTokens
	}
	out := ir.UsageFromTotal(u.PromptTokens, cached+write, u.CompletionTokens, reasoning)
	out.CacheRead, out.CacheWrite = cached, write
	return out
}

func usageObj(u ir.Usage) obj {
	prompt := u.TotalInput()
	o := obj{
		"prompt_tokens":         prompt,
		"completion_tokens":     u.Output,
		"total_tokens":          prompt + u.Output,
		"prompt_tokens_details": obj{"cached_tokens": u.CacheRead},
	}
	if u.CacheWrite > 0 {
		o["prompt_tokens_details"].(obj)["cache_write_tokens"] = u.CacheWrite
	}
	if u.Reasoning > 0 {
		o["completion_tokens_details"] = obj{"reasoning_tokens": u.Reasoning}
	}
	return o
}

type wireChoice struct {
	Index        int          `json:"index"`
	Message      *wireMessage `json:"message"`
	Delta        *wireMessage `json:"delta"`
	FinishReason string       `json:"finish_reason"`
}

type wireResponse struct {
	ID      string       `json:"id"`
	Model   string       `json:"model"`
	Choices []wireChoice `json:"choices"`
	Usage   *wireUsage   `json:"usage"`
	Error   *wireError   `json:"error"`
}

type wireError struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Code    json.RawMessage `json:"code"`
}

func (Codec) DecodeResponse(body []byte) (*ir.Response, error) {
	var w wireResponse
	if err := json.Unmarshal(body, &w); err != nil {
		return nil, fmt.Errorf("invalid chat completions response: %w", err)
	}
	resp := &ir.Response{ID: w.ID, Model: w.Model, StopReason: ir.StopEndTurn}
	if w.Usage != nil {
		resp.Usage = w.Usage.ir()
	}
	if len(w.Choices) == 0 {
		return resp, nil
	}
	c := w.Choices[0]
	resp.StopReason = ir.StopFromOpenAI(c.FinishReason)
	if m := c.Message; m != nil {
		if rc := firstNonEmpty(m.ReasoningContent, m.Reasoning); rc != "" {
			resp.Parts = append(resp.Parts, ir.Part{Type: ir.PartThinking, Text: rc, SigOrigin: self})
		}
		if s, ok := ir.TextOrParts(m.Content); ok && s != "" {
			resp.Parts = append(resp.Parts, ir.Part{Type: ir.PartText, Text: s})
		} else if !ok && len(m.Content) > 0 && string(m.Content) != "null" {
			var ws []wirePart
			if json.Unmarshal(m.Content, &ws) == nil {
				for _, p := range ws {
					if p.Text != "" {
						resp.Parts = append(resp.Parts, ir.Part{Type: ir.PartText, Text: p.Text})
					}
				}
			}
		}
		if m.Refusal != "" {
			resp.Parts = append(resp.Parts, ir.Part{Type: ir.PartText, Text: m.Refusal})
		}
		for _, tc := range m.ToolCalls {
			id := tc.ID
			if id == "" {
				id = ir.NewID("call_")
			}
			resp.Parts = append(resp.Parts, ir.Part{Type: ir.PartToolCall, ToolCallID: id, ToolName: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
	}
	if resp.StopReason == ir.StopEndTurn && resp.HasToolCalls() {
		resp.StopReason = ir.StopToolUse
	}
	return resp, nil
}

func (Codec) EncodeResponse(resp *ir.Response, _ *ir.Request) ([]byte, error) {
	msg := obj{"role": "assistant"}
	var text, thinking strings.Builder
	var calls []obj
	for _, p := range resp.Parts {
		switch p.Type {
		case ir.PartText:
			text.WriteString(p.Text)
		case ir.PartThinking:
			thinking.WriteString(p.Text)
		case ir.PartToolCall:
			args, _ := ir.NormalizeArgs(p.Arguments)
			calls = append(calls, obj{"id": clientToolID(p), "type": "function",
				"function": obj{"name": p.ToolName, "arguments": args}})
		}
	}
	if text.Len() > 0 || len(calls) == 0 {
		msg["content"] = text.String()
	} else {
		msg["content"] = nil
	}
	if thinking.Len() > 0 {
		msg["reasoning_content"] = thinking.String()
	}
	if len(calls) > 0 {
		msg["tool_calls"] = calls
	}
	stop := resp.StopReason
	if len(calls) > 0 && stop == ir.StopEndTurn {
		stop = ir.StopToolUse
	}
	out := obj{
		"id":      chatID(resp.ID),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   resp.Model,
		"choices": []obj{{"index": 0, "message": msg, "finish_reason": ir.StopToOpenAI(stop)}},
		"usage":   usageObj(resp.Usage),
	}
	return json.Marshal(out)
}

func chatID(id string) string {
	if strings.HasPrefix(id, "chatcmpl-") {
		return id
	}
	return ir.NewID("chatcmpl-")
}

func clientToolID(p ir.Part) string {
	id := p.ToolCallID
	if id == "" {
		id = ir.NewID("call_")
	}
	return ir.PackCallID(self, id, p.SigOrigin, p.Signature)
}

// ---- errors ----

func (Codec) DecodeError(status int, body []byte) *ir.APIError {
	var w struct {
		Error *wireError `json:"error"`
	}
	e := &ir.APIError{Status: status, Type: ir.ErrorTypeForStatus(status)}
	if json.Unmarshal(body, &w) == nil && w.Error != nil && w.Error.Message != "" {
		e.Message = w.Error.Message
		return e
	}
	// some providers return {"message": "..."} or a list
	var flat struct {
		Message string `json:"message"`
	}
	if json.Unmarshal(body, &flat) == nil && flat.Message != "" {
		e.Message = flat.Message
		return e
	}
	e.Message = strings.TrimSpace(string(body))
	if e.Message == "" {
		e.Message = fmt.Sprintf("upstream returned status %d", status)
	}
	return e
}

func errorObj(e *ir.APIError) obj {
	typ := e.Type
	switch typ {
	case "authentication_error", "permission_error", "not_found_error", "rate_limit_error", "overloaded_error", "api_error":
	default:
		typ = "invalid_request_error"
		if e.Status >= 500 {
			typ = "api_error"
		}
	}
	return obj{"message": e.Message, "type": typ, "param": nil, "code": nil}
}

func (Codec) EncodeError(e *ir.APIError) []byte {
	b, _ := json.Marshal(obj{"error": errorObj(e)})
	return b
}
