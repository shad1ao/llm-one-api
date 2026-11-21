// Package openaichat implements the OpenAI Chat Completions codec.
package openaichat

import (
	"encoding/json"
	"fmt"
	"strings"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
)

const self = ir.FormatOpenAIChat

func init() { convert.Register(Codec{}) }

// Codec is the OpenAI Chat Completions codec.
type Codec struct{}

func (Codec) Format() ir.Format { return self }

type obj = map[string]any

// ---- wire types ----

type wireFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict"`
	Arguments   string          `json:"arguments"`
}

type wireTool struct {
	Type     string       `json:"type"`
	Function wireFunction `json:"function"`
}

type wireToolCall struct {
	Index    *int         `json:"index"`
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function wireFunction `json:"function"`
}

type wireMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	Name             string          `json:"name"`
	ToolCalls        []wireToolCall  `json:"tool_calls"`
	ToolCallID       string          `json:"tool_call_id"`
	FunctionCall     *wireFunction   `json:"function_call"`
	ReasoningContent string          `json:"reasoning_content"`
	Reasoning        string          `json:"reasoning"`
	Refusal          string          `json:"refusal"`
}

type wirePart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	ImageURL *struct {
		URL    string `json:"url"`
		Detail string `json:"detail"`
	} `json:"image_url"`
	File *struct {
		FileData string `json:"file_data"`
		FileID   string `json:"file_id"`
		Filename string `json:"filename"`
	} `json:"file"`
	InputAudio *struct {
		Data   string `json:"data"`
		Format string `json:"format"`
	} `json:"input_audio"`
}

type wireRequest struct {
	Model               string          `json:"model"`
	Messages            []wireMessage   `json:"messages"`
	Tools               []wireTool      `json:"tools"`
	ToolChoice          json.RawMessage `json:"tool_choice"`
	Functions           []wireFunction  `json:"functions"`
	FunctionCall        json.RawMessage `json:"function_call"`
	ParallelToolCalls   *bool           `json:"parallel_tool_calls"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	TopK                *int            `json:"top_k"`
	Stop                json.RawMessage `json:"stop"`
	Seed                *int64          `json:"seed"`
	Stream              bool            `json:"stream"`
	StreamOptions       *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	ReasoningEffort string `json:"reasoning_effort"`
	Reasoning       *struct {
		Effort    string `json:"effort"`
		MaxTokens int    `json:"max_tokens"`
	} `json:"reasoning"`
	ResponseFormat *struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
			Strict *bool           `json:"strict"`
		} `json:"json_schema"`
	} `json:"response_format"`
	User string `json:"user"`
}

// ---- decode ----

func (Codec) DecodeRequest(body []byte) (*ir.Request, error) {
	var w wireRequest
	if err := json.Unmarshal(body, &w); err != nil {
		return nil, fmt.Errorf("invalid chat completions request: %w", err)
	}
	r := &ir.Request{
		Model:             w.Model,
		Temperature:       w.Temperature,
		TopP:              w.TopP,
		TopK:              w.TopK,
		Seed:              w.Seed,
		Stream:            w.Stream,
		ParallelToolCalls: w.ParallelToolCalls,
		User:              w.User,
	}
	r.MaxTokens = w.MaxCompletionTokens
	if r.MaxTokens == nil {
		r.MaxTokens = w.MaxTokens
	}
	if w.StreamOptions != nil {
		r.IncludeUsage = w.StreamOptions.IncludeUsage
	}
	if s, ok := ir.TextOrParts(w.Stop); ok {
		r.Stop = []string{s}
	} else if len(w.Stop) > 0 {
		_ = json.Unmarshal(w.Stop, &r.Stop)
	}

	legacyIDs := []string{}
	for i, m := range w.Messages {
		switch m.Role {
		case "system", "developer":
			parts, err := decodeContent(m.Content, r)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			for _, p := range parts {
				if p.Type == ir.PartText {
					r.System = append(r.System, p)
				}
			}
		case "user":
			parts, err := decodeContent(m.Content, r)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			r.Messages = append(r.Messages, ir.Message{Role: ir.RoleUser, Parts: parts})
		case "assistant":
			var parts []ir.Part
			if rc := firstNonEmpty(m.ReasoningContent, m.Reasoning); rc != "" {
				parts = append(parts, ir.Part{Type: ir.PartThinking, Text: rc, SigOrigin: self})
			}
			content, err := decodeContent(m.Content, r)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			parts = append(parts, content...)
			if m.Refusal != "" {
				parts = append(parts, ir.Part{Type: ir.PartText, Text: m.Refusal})
			}
			for _, tc := range m.ToolCalls {
				id, origin, sig := ir.UnpackCallID(tc.ID)
				parts = append(parts, ir.Part{Type: ir.PartToolCall, ToolCallID: id, ToolName: tc.Function.Name,
					Arguments: tc.Function.Arguments, Signature: sig, SigOrigin: origin})
			}
			if fc := m.FunctionCall; fc != nil {
				id := fmt.Sprintf("call_legacy_%d", len(legacyIDs))
				legacyIDs = append(legacyIDs, id)
				parts = append(parts, ir.Part{Type: ir.PartToolCall, ToolCallID: id, ToolName: fc.Name, Arguments: fc.Arguments})
			}
			r.Messages = append(r.Messages, ir.Message{Role: ir.RoleAssistant, Parts: parts})
		case "tool", "function":
			content, err := decodeContent(m.Content, r)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			id, _, _ := ir.UnpackCallID(m.ToolCallID)
			if m.Role == "function" && len(legacyIDs) > 0 {
				id = legacyIDs[len(legacyIDs)-1]
			}
			r.Messages = append(r.Messages, ir.Message{Role: ir.RoleUser, Parts: []ir.Part{{
				Type: ir.PartToolResult, ToolCallID: id, ToolName: m.Name, Content: content}}})
		default:
			r.Warn("unknown_role", "unknown message role "+m.Role)
		}
	}
	r.Messages = ir.MergeAdjacent(r.Messages)

	for _, t := range w.Tools {
		if t.Type != "function" && t.Type != "" {
			r.Warn("hosted_tool_dropped", "tool type "+t.Type+" is not portable")
			continue
		}
		r.Tools = append(r.Tools, ir.Tool{Name: t.Function.Name, Description: t.Function.Description,
			Parameters: ir.SchemaOrEmpty(t.Function.Parameters), Strict: t.Function.Strict})
	}
	for _, f := range w.Functions {
		r.Tools = append(r.Tools, ir.Tool{Name: f.Name, Description: f.Description, Parameters: ir.SchemaOrEmpty(f.Parameters)})
	}

	choice := w.ToolChoice
	if len(choice) == 0 {
		choice = w.FunctionCall
	}
	r.ToolChoice = decodeToolChoice(choice)

	switch {
	case w.ReasoningEffort != "":
		r.Reasoning = &ir.Reasoning{Effort: w.ReasoningEffort}
	case w.Reasoning != nil && (w.Reasoning.Effort != "" || w.Reasoning.MaxTokens > 0):
		r.Reasoning = &ir.Reasoning{Effort: w.Reasoning.Effort, BudgetTokens: w.Reasoning.MaxTokens}
	}

	if rf := w.ResponseFormat; rf != nil && rf.Type != "" {
		f := &ir.ResponseFormat{Type: rf.Type}
		if rf.JSONSchema != nil {
			f.Name, f.Schema, f.Strict = rf.JSONSchema.Name, rf.JSONSchema.Schema, rf.JSONSchema.Strict
		}
		r.ResponseFormat = f
	}
	return r, nil
}

func decodeToolChoice(raw json.RawMessage) *ir.ToolChoice {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if s, ok := ir.TextOrParts(raw); ok {
		switch s {
		case "auto":
			return &ir.ToolChoice{Mode: ir.ToolChoiceAuto}
		case "none":
			return &ir.ToolChoice{Mode: ir.ToolChoiceNone}
		case "required", "any":
			return &ir.ToolChoice{Mode: ir.ToolChoiceRequired}
		}
		return nil
	}
	var o struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &o) != nil {
		return nil
	}
	name := firstNonEmpty(o.Function.Name, o.Name)
	if name == "" {
		return &ir.ToolChoice{Mode: ir.ToolChoiceAuto}
	}
	return &ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: name}
}

func decodeContent(raw json.RawMessage, r *ir.Request) ([]ir.Part, error) {
	if s, ok := ir.TextOrParts(raw); ok {
		if s == "" {
			return nil, nil
		}
		return []ir.Part{{Type: ir.PartText, Text: s}}, nil
	}
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var ws []wirePart
	if err := json.Unmarshal(raw, &ws); err != nil {
		return nil, err
	}
	var parts []ir.Part
	for _, w := range ws {
		switch w.Type {
		case "text", "input_text", "output_text":
			parts = append(parts, ir.Part{Type: ir.PartText, Text: w.Text})
		case "image_url":
			if w.ImageURL != nil {
				m := ir.MediaFromURL(w.ImageURL.URL)
				m.Detail = w.ImageURL.Detail
				parts = append(parts, ir.Part{Type: ir.PartImage, Media: m})
			}
		case "file":
			if w.File == nil {
				continue
			}
			if w.File.FileData == "" {
				r.Warn("unsupported_media_source", "file_id references are not portable")
				continue
			}
			m := ir.MediaFromURL(w.File.FileData)
			if m.URL != "" { // raw base64 without data: prefix
				m = &ir.Media{MediaType: "application/pdf", Data: w.File.FileData}
			}
			m.Filename = w.File.Filename
			parts = append(parts, ir.Part{Type: ir.PartFile, Media: m})
		case "refusal":
			parts = append(parts, ir.Part{Type: ir.PartText, Text: w.Text})
		default:
			r.Warn("unsupported_content", "content part "+w.Type+" dropped")
		}
	}
	return parts, nil
}

func firstNonEmpty(ss ...string) string {
	for _, s := range ss {
		if s != "" {
			return s
		}
	}
	return ""
}

// ---- encode ----

// isOpenAIReasoningModel matches models that require max_completion_tokens
// and reject sampling parameters.
func isOpenAIReasoningModel(model string) bool {
	m := strings.ToLower(model)
	if i := strings.LastIndex(m, "/"); i >= 0 {
		m = m[i+1:]
	}
	for _, p := range []string{"o1", "o3", "o4", "gpt-5"} {
		if strings.HasPrefix(m, p) {
			return !strings.HasPrefix(m, "gpt-5-chat")
		}
	}
	return false
}

func (Codec) EncodeRequest(r *ir.Request) ([]byte, error) {
	out := obj{"model": r.Model}
	reasoningModel := isOpenAIReasoningModel(r.Model)

	var msgs []obj
	if sys := joinText(r.System); sys != "" {
		msgs = append(msgs, obj{"role": "system", "content": sys})
	}
	for _, m := range ir.RepairToolPairing(r.Messages, &r.Diags) {
		if m.Role == ir.RoleAssistant {
			if am := encodeAssistant(m.Parts, r); am != nil {
				msgs = append(msgs, am)
			}
			continue
		}
		msgs = append(msgs, encodeUser(m.Parts, r)...)
	}
	out["messages"] = msgs

	if len(r.Tools) > 0 {
		tools := make([]obj, 0, len(r.Tools))
		for _, t := range r.Tools {
			fn := obj{"name": t.Name, "parameters": ir.SchemaOrEmpty(t.Parameters)}
			if t.Description != "" {
				fn["description"] = t.Description
			}
			if t.Strict != nil {
				fn["strict"] = *t.Strict
			}
			tools = append(tools, obj{"type": "function", "function": fn})
		}
		out["tools"] = tools
		if tc := r.ToolChoice; tc != nil {
			switch tc.Mode {
			case ir.ToolChoiceTool:
				out["tool_choice"] = obj{"type": "function", "function": obj{"name": tc.Name}}
			default:
				out["tool_choice"] = string(tc.Mode)
			}
		}
		if r.ParallelToolCalls != nil {
			out["parallel_tool_calls"] = *r.ParallelToolCalls
		}
	}

	if r.MaxTokens != nil {
		if reasoningModel {
			out["max_completion_tokens"] = *r.MaxTokens
		} else {
			out["max_tokens"] = *r.MaxTokens
		}
	}
	if reasoningModel {
		if r.Temperature != nil || r.TopP != nil {
			r.Info("sampling_dropped", "temperature/top_p dropped for reasoning model")
		}
	} else {
		if r.Temperature != nil {
			out["temperature"] = *r.Temperature
		}
		if r.TopP != nil {
			out["top_p"] = *r.TopP
		}
	}
	if r.TopK != nil {
		r.Info("top_k_dropped", "top_k is not supported by chat completions")
	}
	if len(r.Stop) > 0 {
		out["stop"] = r.Stop
	}
	if r.Seed != nil {
		out["seed"] = *r.Seed
	}
	if r.User != "" {
		out["user"] = r.User
	}
	if r.Stream {
		out["stream"] = true
		out["stream_options"] = obj{"include_usage": true}
	}
	if rs := r.Reasoning; rs != nil {
		if e := rs.EffortLevel(); e != "" {
			out["reasoning_effort"] = e
		}
	}
	if rf := r.ResponseFormat; rf != nil && rf.Type != "" {
		f := obj{"type": rf.Type}
		if rf.Type == "json_schema" {
			js := obj{"name": firstNonEmpty(rf.Name, "response"), "schema": rf.Schema}
			if rf.Strict != nil {
				js["strict"] = *rf.Strict
			}
			f["json_schema"] = js
		}
		out["response_format"] = f
	}
	return json.Marshal(out)
}

func joinText(parts []ir.Part) string {
	var texts []string
	for _, p := range parts {
		if p.Type == ir.PartText && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "\n\n")
}

// encodeUser emits tool messages for tool results (first, as the API
// requires) and then a user message with the remaining content. Images
// returned by tools are lifted into the user message because tool messages
// only carry text.
func encodeUser(parts []ir.Part, r *ir.Request) []obj {
	var out []obj
	var rest []ir.Part
	var lifted []ir.Part
	for _, p := range parts {
		if p.Type != ir.PartToolResult {
			rest = append(rest, p)
			continue
		}
		text := ir.TextOf(p.Content)
		hasMedia := false
		for _, c := range p.Content {
			if c.Type == ir.PartImage || c.Type == ir.PartFile {
				lifted = append(lifted, c)
				hasMedia = true
			}
		}
		if text == "" {
			if hasMedia {
				text = "[image]"
			} else {
				text = "(empty)"
			}
		}
		if p.IsError {
			text = "Error: " + text
		}
		out = append(out, obj{"role": "tool", "tool_call_id": p.ToolCallID, "content": text})
	}
	if len(lifted) > 0 {
		rest = append([]ir.Part{{Type: ir.PartText, Text: "Images returned by the tool call:"}}, append(lifted, rest...)...)
	}
	if c := encodeUserContent(rest, r); c != nil {
		out = append(out, obj{"role": "user", "content": c})
	}
	return out
}

func encodeUserContent(parts []ir.Part, r *ir.Request) any {
	if len(parts) == 0 {
		return nil
	}
	allText := true
	for _, p := range parts {
		if p.Type != ir.PartText {
			allText = false
		}
	}
	if allText {
		return joinTextKeep(parts)
	}
	var arr []obj
	for _, p := range parts {
		switch p.Type {
		case ir.PartText:
			if p.Text != "" {
				arr = append(arr, obj{"type": "text", "text": p.Text})
			}
		case ir.PartImage:
			iu := obj{"url": p.Media.URLOf()}
			if p.Media.Detail != "" {
				iu["detail"] = p.Media.Detail
			}
			arr = append(arr, obj{"type": "image_url", "image_url": iu})
		case ir.PartFile:
			if p.Media.Data == "" {
				r.Warn("unsupported_media_source", "chat completions files require inline data; url dropped")
				continue
			}
			f := obj{"file_data": p.Media.URLOf()}
			f["filename"] = firstNonEmpty(p.Media.Filename, "file.pdf")
			arr = append(arr, obj{"type": "file", "file": f})
		}
	}
	if len(arr) == 0 {
		return nil
	}
	return arr
}

func joinTextKeep(parts []ir.Part) any {
	var b strings.Builder
	for i, p := range parts {
		if i > 0 && b.Len() > 0 && p.Text != "" {
			b.WriteString("\n")
		}
		b.WriteString(p.Text)
	}
	if b.Len() == 0 {
		return nil
	}
	return b.String()
}

func encodeAssistant(parts []ir.Part, r *ir.Request) obj {
	m := obj{"role": "assistant"}
	var text strings.Builder
	var thinking strings.Builder
	var calls []obj
	for _, p := range parts {
		switch p.Type {
		case ir.PartText:
			text.WriteString(p.Text)
		case ir.PartThinking:
			thinking.WriteString(p.Text)
		case ir.PartToolCall:
			args, ok := ir.NormalizeArgs(p.Arguments)
			if !ok {
				r.Warn("invalid_tool_arguments", "tool call "+p.ToolCallID+" has non-object arguments")
			}
			calls = append(calls, obj{"id": p.ToolCallID, "type": "function",
				"function": obj{"name": p.ToolName, "arguments": args}})
		case ir.PartImage, ir.PartFile:
			r.Info("assistant_media_dropped", "assistant media is not supported by chat completions")
		}
	}
	if text.Len() == 0 && len(calls) == 0 {
		return nil
	}
	if text.Len() > 0 {
		m["content"] = text.String()
	} else {
		m["content"] = nil
	}
	if len(calls) > 0 {
		m["tool_calls"] = calls
		// DeepSeek-style thinking models require reasoning_content to be
		// replayed on tool-call turns.
		if thinking.Len() > 0 {
			m["reasoning_content"] = thinking.String()
		}
	}
	return m
}
