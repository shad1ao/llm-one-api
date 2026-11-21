package gemini

import (
	"encoding/json"
	"fmt"
	"strings"

	"llm-one-api/internal/convert/ir"
)

type wireUsage struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	ToolUsePromptTokenCount int `json:"toolUsePromptTokenCount"`
}

func (u wireUsage) ir() ir.Usage {
	return ir.UsageFromTotal(u.PromptTokenCount+u.ToolUsePromptTokenCount, u.CachedContentTokenCount,
		u.CandidatesTokenCount+u.ThoughtsTokenCount, u.ThoughtsTokenCount)
}

func usageObj(u ir.Usage) obj {
	prompt := u.TotalInput()
	o := obj{
		"promptTokenCount":     prompt,
		"candidatesTokenCount": u.Output - u.Reasoning,
		"totalTokenCount":      prompt + u.Output,
	}
	if u.Reasoning > 0 {
		o["thoughtsTokenCount"] = u.Reasoning
	}
	if u.CacheRead > 0 {
		o["cachedContentTokenCount"] = u.CacheRead
	}
	return o
}

type wireCandidate struct {
	Content      *wireContent `json:"content"`
	FinishReason string       `json:"finishReason"`
	Index        int          `json:"index"`
}

type wireResponse struct {
	Candidates     []wireCandidate `json:"candidates"`
	UsageMetadata  *wireUsage      `json:"usageMetadata"`
	ModelVersion   string          `json:"modelVersion"`
	ResponseID     string          `json:"responseId"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	Error *wireError `json:"error"`
}

type wireError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// partsFrom converts response parts. idGen supplies ids for function calls.
func partsFrom(ps []wirePart, idGen func() string) []ir.Part {
	var out []ir.Part
	for _, p := range ps {
		switch {
		case p.FunctionCall != nil:
			id := p.FunctionCall.ID
			if id == "" {
				id = idGen()
			}
			out = append(out, ir.Part{Type: ir.PartToolCall, ToolCallID: id, ToolName: p.FunctionCall.Name,
				Arguments: ir.RawString(p.FunctionCall.Args), Signature: p.ThoughtSignature, SigOrigin: self})
		case p.Text != nil && p.Thought:
			out = append(out, ir.Part{Type: ir.PartThinking, Text: *p.Text, Signature: p.ThoughtSignature, SigOrigin: self})
		case p.Text != nil:
			if *p.Text != "" {
				out = append(out, ir.Part{Type: ir.PartText, Text: *p.Text})
			}
		}
	}
	return out
}

func newIDGen() func() string { return func() string { return ir.NewID("call_") } }

func (Codec) DecodeResponse(body []byte) (*ir.Response, error) {
	var w wireResponse
	if err := json.Unmarshal(body, &w); err != nil {
		return nil, fmt.Errorf("invalid gemini response: %w", err)
	}
	resp := &ir.Response{ID: w.ResponseID, Model: w.ModelVersion, StopReason: ir.StopEndTurn}
	if w.UsageMetadata != nil {
		resp.Usage = w.UsageMetadata.ir()
	}
	if len(w.Candidates) == 0 {
		if w.PromptFeedback != nil && w.PromptFeedback.BlockReason != "" {
			resp.StopReason = ir.StopContentFilter
		}
		return resp, nil
	}
	c := w.Candidates[0]
	if c.Content != nil {
		resp.Parts = partsFrom(c.Content.Parts, newIDGen())
	}
	resp.StopReason = ir.StopFromGemini(c.FinishReason)
	if resp.HasToolCalls() && resp.StopReason == ir.StopEndTurn {
		resp.StopReason = ir.StopToolUse
	}
	return resp, nil
}

func clientParts(parts []ir.Part) []obj {
	var out []obj
	for _, p := range parts {
		switch p.Type {
		case ir.PartText:
			out = append(out, obj{"text": p.Text})
		case ir.PartThinking:
			o := obj{"text": p.Text, "thought": true}
			if s := sigToClient(p.SigOrigin, p.Signature); s != "" {
				o["thoughtSignature"] = s
			}
			out = append(out, o)
		case ir.PartToolCall:
			args, _ := ir.NormalizeArgs(p.Arguments)
			if !json.Valid([]byte(args)) {
				args = "{}"
			}
			o := obj{"functionCall": obj{"id": p.ToolCallID, "name": p.ToolName, "args": json.RawMessage(args)}}
			if s := sigToClient(p.SigOrigin, p.Signature); s != "" {
				o["thoughtSignature"] = s
			}
			out = append(out, o)
		}
	}
	return out
}

func finishReason(stop ir.StopReason) string { return ir.StopToGemini(stop) }

func (Codec) EncodeResponse(resp *ir.Response, _ *ir.Request) ([]byte, error) {
	parts := clientParts(resp.Parts)
	if parts == nil {
		parts = []obj{{"text": ""}}
	}
	out := obj{
		"candidates": []obj{{"content": obj{"role": "model", "parts": parts},
			"finishReason": finishReason(resp.StopReason), "index": 0}},
		"usageMetadata": usageObj(resp.Usage),
		"modelVersion":  resp.Model,
		"responseId":    firstNonEmpty(resp.ID, ir.NewID("")),
	}
	return json.Marshal(out)
}

func firstNonEmpty(ss ...string) string {
	for _, s := range ss {
		if s != "" {
			return s
		}
	}
	return ""
}

// ---- errors ----

var statusNames = map[int]string{
	400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
	409: "ABORTED", 413: "INVALID_ARGUMENT", 429: "RESOURCE_EXHAUSTED", 499: "CANCELLED",
	500: "INTERNAL", 501: "UNIMPLEMENTED", 502: "UNAVAILABLE", 503: "UNAVAILABLE", 504: "DEADLINE_EXCEEDED", 529: "UNAVAILABLE",
}

func (Codec) DecodeError(status int, body []byte) *ir.APIError {
	e := &ir.APIError{Status: status, Type: ir.ErrorTypeForStatus(status)}
	trimmed := strings.TrimSpace(string(body))
	var w struct {
		Error *wireError `json:"error"`
	}
	if strings.HasPrefix(trimmed, "[") {
		var arr []json.RawMessage
		if json.Unmarshal(body, &arr) == nil && len(arr) > 0 {
			body = arr[0]
		}
	}
	if json.Unmarshal(body, &w) == nil && w.Error != nil && w.Error.Message != "" {
		e.Message = w.Error.Message
		return e
	}
	e.Message = trimmed
	if e.Message == "" {
		e.Message = fmt.Sprintf("upstream returned status %d", status)
	}
	return e
}

func errorObj(e *ir.APIError) obj {
	status := e.Status
	if status == 0 {
		status = 500
	}
	name := statusNames[status]
	if name == "" {
		name = "UNKNOWN"
	}
	return obj{"code": status, "message": e.Message, "status": name}
}

func (Codec) EncodeError(e *ir.APIError) []byte {
	b, _ := json.Marshal(obj{"error": errorObj(e)})
	return b
}
