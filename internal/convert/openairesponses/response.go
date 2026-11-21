package openairesponses

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"llm-one-api/internal/convert/ir"
)

type wireUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

func (u wireUsage) ir() ir.Usage {
	cached, reasoning := 0, 0
	if u.InputTokensDetails != nil {
		cached = u.InputTokensDetails.CachedTokens
	}
	if u.OutputTokensDetails != nil {
		reasoning = u.OutputTokensDetails.ReasoningTokens
	}
	return ir.UsageFromTotal(u.InputTokens, cached, u.OutputTokens, reasoning)
}

func usageObj(u ir.Usage) obj {
	in := u.TotalInput()
	return obj{
		"input_tokens":          in,
		"input_tokens_details":  obj{"cached_tokens": u.CacheRead},
		"output_tokens":         u.Output,
		"output_tokens_details": obj{"reasoning_tokens": u.Reasoning},
		"total_tokens":          in + u.Output,
	}
}

type wireResponse struct {
	ID                string     `json:"id"`
	Model             string     `json:"model"`
	Status            string     `json:"status"`
	Output            []wireItem `json:"output"`
	Usage             *wireUsage `json:"usage"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func stopFrom(status, incompleteReason string, hasTool bool) ir.StopReason {
	if status == "incomplete" {
		switch incompleteReason {
		case "max_output_tokens":
			return ir.StopMaxTokens
		case "content_filter":
			return ir.StopContentFilter
		}
		return ir.StopMaxTokens
	}
	if hasTool {
		return ir.StopToolUse
	}
	return ir.StopEndTurn
}

func (w *wireResponse) toIR() *ir.Response {
	resp := &ir.Response{ID: w.ID, Model: w.Model}
	if w.Usage != nil {
		resp.Usage = w.Usage.ir()
	}
	for _, it := range w.Output {
		resp.Parts = append(resp.Parts, outputItemParts(it)...)
	}
	reason := ""
	if w.IncompleteDetails != nil {
		reason = w.IncompleteDetails.Reason
	}
	resp.StopReason = stopFrom(w.Status, reason, resp.HasToolCalls())
	return resp
}

func outputItemParts(it wireItem) []ir.Part {
	switch it.Type {
	case "reasoning":
		var texts []string
		for _, s := range it.Summary {
			texts = append(texts, s.Text)
		}
		// Some providers return full reasoning text in content.
		if len(texts) == 0 && len(it.Content) > 0 {
			var cs []wireContent
			if json.Unmarshal(it.Content, &cs) == nil {
				for _, c := range cs {
					texts = append(texts, c.Text)
				}
			}
		}
		return []ir.Part{{Type: ir.PartThinking, Text: strings.Join(texts, "\n\n"),
			Signature: packReasoning(it.ID, it.EncryptedContent), SigOrigin: self}}
	case "message":
		var cs []wireContent
		_ = json.Unmarshal(it.Content, &cs)
		var parts []ir.Part
		for _, c := range cs {
			switch c.Type {
			case "output_text":
				parts = append(parts, ir.Part{Type: ir.PartText, Text: c.Text})
			case "refusal":
				parts = append(parts, ir.Part{Type: ir.PartText, Text: c.Refusal})
			}
		}
		return parts
	case "function_call":
		return []ir.Part{{Type: ir.PartToolCall, ToolCallID: it.CallID, ToolName: it.Name, Arguments: it.Arguments}}
	case "custom_tool_call":
		args, _ := json.Marshal(obj{"input": it.Input})
		return []ir.Part{{Type: ir.PartToolCall, ToolCallID: it.CallID, ToolName: it.Name, Arguments: string(args)}}
	}
	return nil
}

func (Codec) DecodeResponse(body []byte) (*ir.Response, error) {
	var w wireResponse
	if err := json.Unmarshal(body, &w); err != nil {
		return nil, fmt.Errorf("invalid responses response: %w", err)
	}
	if w.Status == "failed" && w.Error != nil {
		return nil, &ir.APIError{Status: 502, Type: "api_error", Message: w.Error.Message}
	}
	return w.toIR(), nil
}

// ---- encode ----

func respID(id string) string {
	if strings.HasPrefix(id, "resp_") {
		return id
	}
	return ir.NewID("resp_")
}

func reasoningItem(p ir.Part) obj {
	id, enc := "", ""
	if p.SigOrigin == self {
		id, enc = unpackReasoning(p.Signature)
	} else {
		enc = ir.SigFor(self, p.SigOrigin, p.Signature)
	}
	if id == "" {
		id = ir.NewID("rs_")
	}
	item := obj{"id": id, "type": "reasoning", "summary": []obj{}}
	if p.Text != "" {
		item["summary"] = []obj{{"type": "summary_text", "text": p.Text}}
	}
	if enc != "" {
		item["encrypted_content"] = enc
	}
	return item
}

func messageItem(id, text, status string) obj {
	return obj{"id": id, "type": "message", "status": status, "role": "assistant",
		"content": []obj{{"type": "output_text", "text": text, "annotations": []any{}}}}
}

func callItem(p ir.Part, req *ir.Request, status string) obj {
	callID := ir.PackCallID(self, p.ToolCallID, p.SigOrigin, p.Signature)
	if t := req.ToolByName(p.ToolName); t != nil && t.Custom {
		var a struct {
			Input string `json:"input"`
		}
		_ = json.Unmarshal([]byte(p.Arguments), &a)
		return obj{"id": ir.NewID("ctc_"), "type": "custom_tool_call", "status": status, "call_id": callID, "name": p.ToolName, "input": a.Input}
	}
	args, _ := ir.NormalizeArgs(p.Arguments)
	return obj{"id": ir.NewID("fc_"), "type": "function_call", "status": status, "call_id": callID, "name": p.ToolName, "arguments": args}
}

func statusOf(stop ir.StopReason) (string, any) {
	switch stop {
	case ir.StopMaxTokens, ir.StopPause:
		return "incomplete", obj{"reason": "max_output_tokens"}
	case ir.StopContentFilter:
		return "incomplete", obj{"reason": "content_filter"}
	}
	return "completed", nil
}

func responseEnvelope(id, model, status string, output []obj) obj {
	return obj{
		"id": id, "object": "response", "created_at": time.Now().Unix(), "status": status,
		"model": model, "output": output, "error": nil, "incomplete_details": nil,
		"instructions": nil, "metadata": obj{}, "parallel_tool_calls": true,
		"tool_choice": "auto", "tools": []any{}, "temperature": nil, "top_p": nil,
	}
}

func (Codec) EncodeResponse(resp *ir.Response, req *ir.Request) ([]byte, error) {
	var output []obj
	var text strings.Builder
	flush := func() {
		if text.Len() > 0 {
			output = append(output, messageItem(ir.NewID("msg_"), text.String(), "completed"))
			text.Reset()
		}
	}
	for _, p := range resp.Parts {
		switch p.Type {
		case ir.PartText:
			text.WriteString(p.Text)
		case ir.PartThinking, ir.PartRedactedThinking:
			flush()
			output = append(output, reasoningItem(p))
		case ir.PartToolCall:
			flush()
			output = append(output, callItem(p, req, "completed"))
		}
	}
	flush()
	if output == nil {
		output = []obj{}
	}
	status, incomplete := statusOf(resp.StopReason)
	o := responseEnvelope(respID(resp.ID), resp.Model, status, output)
	o["incomplete_details"] = incomplete
	o["usage"] = usageObj(resp.Usage)
	return json.Marshal(o)
}

// ---- errors ----

func (Codec) DecodeError(status int, body []byte) *ir.APIError {
	var w struct {
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	e := &ir.APIError{Status: status, Type: ir.ErrorTypeForStatus(status)}
	if json.Unmarshal(body, &w) == nil && w.Error != nil && w.Error.Message != "" {
		e.Message = w.Error.Message
		return e
	}
	e.Message = strings.TrimSpace(string(body))
	if e.Message == "" {
		e.Message = fmt.Sprintf("upstream returned status %d", status)
	}
	return e
}

func errType(e *ir.APIError) string {
	if e.Status >= 500 {
		return "server_error"
	}
	return "invalid_request_error"
}

func (Codec) EncodeError(e *ir.APIError) []byte {
	b, _ := json.Marshal(obj{"error": obj{"message": e.Message, "type": errType(e), "param": nil, "code": nil}})
	return b
}
