// Package openairesponses implements the OpenAI Responses API codec.
package openairesponses

import (
	"encoding/json"
	"fmt"
	"strings"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
)

const self = ir.FormatOpenAIResponses

func init() { convert.Register(Codec{}) }

// Codec is the OpenAI Responses codec.
type Codec struct{}

func (Codec) Format() ir.Format { return self }

type obj = map[string]any

// Reasoning items are replayed with both their id and encrypted content; we
// pack them into a single signature string.
func packReasoning(id, enc string) string {
	if enc == "" {
		return ""
	}
	return id + "|" + enc
}

func unpackReasoning(sig string) (id, enc string) {
	if id, enc, ok := strings.Cut(sig, "|"); ok && strings.HasPrefix(id, "rs_") {
		return id, enc
	}
	return "", sig
}

// ---- wire types ----

type wireContent struct {
	Type     string          `json:"type"`
	Text     string          `json:"text"`
	Refusal  string          `json:"refusal"`
	ImageURL string          `json:"image_url"`
	FileID   string          `json:"file_id"`
	FileData string          `json:"file_data"`
	FileURL  string          `json:"file_url"`
	Filename string          `json:"filename"`
	Detail   string          `json:"detail"`
	Raw      json.RawMessage `json:"-"`
}

type wireItem struct {
	Type             string          `json:"type"`
	ID               string          `json:"id"`
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	CallID           string          `json:"call_id"`
	Name             string          `json:"name"`
	Arguments        string          `json:"arguments"`
	Input            string          `json:"input"`
	Output           json.RawMessage `json:"output"`
	Summary          []wireContent   `json:"summary"`
	EncryptedContent string          `json:"encrypted_content"`
	Status           string          `json:"status"`
}

type wireTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict"`
}

type wireRequest struct {
	Model              string          `json:"model"`
	Instructions       string          `json:"instructions"`
	Input              json.RawMessage `json:"input"`
	Tools              []wireTool      `json:"tools"`
	ToolChoice         json.RawMessage `json:"tool_choice"`
	ParallelToolCalls  *bool           `json:"parallel_tool_calls"`
	MaxOutputTokens    *int            `json:"max_output_tokens"`
	Temperature        *float64        `json:"temperature"`
	TopP               *float64        `json:"top_p"`
	Stream             bool            `json:"stream"`
	PreviousResponseID string          `json:"previous_response_id"`
	User               string          `json:"user"`
	Reasoning          *struct {
		Effort  string `json:"effort"`
		Summary string `json:"summary"`
	} `json:"reasoning"`
	Text *struct {
		Format *struct {
			Type   string          `json:"type"`
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
			Strict *bool           `json:"strict"`
		} `json:"format"`
	} `json:"text"`
}

// ---- decode ----

func (Codec) DecodeRequest(body []byte) (*ir.Request, error) {
	var w wireRequest
	if err := json.Unmarshal(body, &w); err != nil {
		return nil, fmt.Errorf("invalid responses request: %w", err)
	}
	r := &ir.Request{
		Model:             w.Model,
		MaxTokens:         w.MaxOutputTokens,
		Temperature:       w.Temperature,
		TopP:              w.TopP,
		Stream:            w.Stream,
		IncludeUsage:      true,
		ParallelToolCalls: w.ParallelToolCalls,
		User:              w.User,
	}
	if w.Instructions != "" {
		r.System = append(r.System, ir.Part{Type: ir.PartText, Text: w.Instructions})
	}
	if w.PreviousResponseID != "" {
		r.Diags.Add(ir.SevError, "previous_response_id", "previous_response_id cannot be honoured by a non-Responses upstream; history is lost")
	}

	customTools := map[string]bool{}
	for _, t := range w.Tools {
		switch t.Type {
		case "function":
			r.Tools = append(r.Tools, ir.Tool{Name: t.Name, Description: t.Description,
				Parameters: ir.SchemaOrEmpty(t.Parameters), Strict: t.Strict})
		case "custom":
			customTools[t.Name] = true
			r.Tools = append(r.Tools, ir.Tool{Name: t.Name, Description: t.Description, Custom: true,
				Parameters: json.RawMessage(`{"type":"object","properties":{"input":{"type":"string","description":"Raw tool input"}},"required":["input"]}`)})
		default:
			r.Warn("hosted_tool_dropped", "responses tool type "+t.Type+" is not portable")
		}
	}

	if s, ok := ir.TextOrParts(w.Input); ok {
		r.Messages = []ir.Message{{Role: ir.RoleUser, Parts: []ir.Part{{Type: ir.PartText, Text: s}}}}
	} else if len(w.Input) > 0 && string(w.Input) != "null" {
		var items []wireItem
		if err := json.Unmarshal(w.Input, &items); err != nil {
			return nil, fmt.Errorf("invalid responses input: %w", err)
		}
		for _, it := range items {
			if err := decodeItem(it, r); err != nil {
				return nil, err
			}
		}
		r.Messages = ir.MergeAdjacent(r.Messages)
	}

	r.ToolChoice = decodeToolChoice(w.ToolChoice)
	if rs := w.Reasoning; rs != nil && rs.Effort != "" {
		r.Reasoning = &ir.Reasoning{Effort: rs.Effort, Summary: rs.Summary}
	}
	if w.Text != nil && w.Text.Format != nil && w.Text.Format.Type != "" && w.Text.Format.Type != "text" {
		f := w.Text.Format
		r.ResponseFormat = &ir.ResponseFormat{Type: f.Type, Name: f.Name, Schema: f.Schema, Strict: f.Strict}
	}
	return r, nil
}

func push(r *ir.Request, role ir.Role, parts ...ir.Part) {
	r.Messages = append(r.Messages, ir.Message{Role: role, Parts: parts})
}

func decodeItem(it wireItem, r *ir.Request) error {
	typ := it.Type
	if typ == "" && it.Role != "" {
		typ = "message"
	}
	switch typ {
	case "message":
		parts, err := decodeContent(it.Content, r)
		if err != nil {
			return err
		}
		switch it.Role {
		case "system", "developer":
			for _, p := range parts {
				if p.Type == ir.PartText {
					r.System = append(r.System, p)
				}
			}
		case "assistant":
			push(r, ir.RoleAssistant, parts...)
		default:
			push(r, ir.RoleUser, parts...)
		}
	case "function_call":
		id, origin, sig := ir.UnpackCallID(it.CallID)
		push(r, ir.RoleAssistant, ir.Part{Type: ir.PartToolCall, ToolCallID: id, ToolName: it.Name,
			Arguments: it.Arguments, Signature: sig, SigOrigin: origin})
	case "custom_tool_call":
		id, origin, sig := ir.UnpackCallID(it.CallID)
		args, _ := json.Marshal(obj{"input": it.Input})
		push(r, ir.RoleAssistant, ir.Part{Type: ir.PartToolCall, ToolCallID: id, ToolName: it.Name,
			Arguments: string(args), Signature: sig, SigOrigin: origin})
	case "function_call_output", "custom_tool_call_output":
		id, _, _ := ir.UnpackCallID(it.CallID)
		p := ir.Part{Type: ir.PartToolResult, ToolCallID: id}
		if s, ok := ir.TextOrParts(it.Output); ok {
			p.Content = []ir.Part{{Type: ir.PartText, Text: s}}
		} else {
			parts, err := decodeContent(it.Output, r)
			if err != nil {
				return err
			}
			p.Content = parts
		}
		push(r, ir.RoleUser, p)
	case "reasoning":
		var texts []string
		for _, s := range it.Summary {
			texts = append(texts, s.Text)
		}
		origin, sig := ir.UnwrapSig(it.EncryptedContent, self)
		if origin == self {
			sig = packReasoning(it.ID, sig)
		}
		push(r, ir.RoleAssistant, ir.Part{Type: ir.PartThinking, Text: strings.Join(texts, "\n\n"), Signature: sig, SigOrigin: origin})
	case "item_reference":
		r.Diags.Add(ir.SevError, "item_reference", "item_reference requires server-side state")
	default:
		r.Info("item_dropped", "responses input item "+typ+" dropped")
	}
	return nil
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
	var cs []wireContent
	if err := json.Unmarshal(raw, &cs); err != nil {
		return nil, err
	}
	var parts []ir.Part
	for _, c := range cs {
		switch c.Type {
		case "input_text", "output_text", "text", "summary_text":
			parts = append(parts, ir.Part{Type: ir.PartText, Text: c.Text})
		case "refusal":
			parts = append(parts, ir.Part{Type: ir.PartText, Text: c.Refusal})
		case "input_image":
			if c.ImageURL == "" {
				r.Warn("unsupported_media_source", "input_image file_id is not portable")
				continue
			}
			m := ir.MediaFromURL(c.ImageURL)
			m.Detail = c.Detail
			parts = append(parts, ir.Part{Type: ir.PartImage, Media: m})
		case "input_file":
			var m *ir.Media
			switch {
			case c.FileData != "":
				m = ir.MediaFromURL(c.FileData)
				if m.URL != "" {
					m = &ir.Media{MediaType: "application/pdf", Data: c.FileData}
				}
			case c.FileURL != "":
				m = &ir.Media{URL: c.FileURL}
			default:
				r.Warn("unsupported_media_source", "input_file file_id is not portable")
				continue
			}
			m.Filename = c.Filename
			parts = append(parts, ir.Part{Type: ir.PartFile, Media: m})
		default:
			r.Warn("unsupported_content", "responses content "+c.Type+" dropped")
		}
	}
	return parts, nil
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
		case "required":
			return &ir.ToolChoice{Mode: ir.ToolChoiceRequired}
		}
		return nil
	}
	var o struct {
		Type string `json:"type"`
		Name string `json:"name"`
		Mode string `json:"mode"`
	}
	if json.Unmarshal(raw, &o) != nil {
		return nil
	}
	if (o.Type == "function" || o.Type == "custom") && o.Name != "" {
		return &ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: o.Name}
	}
	if o.Type == "allowed_tools" && o.Mode == "required" {
		return &ir.ToolChoice{Mode: ir.ToolChoiceRequired}
	}
	return &ir.ToolChoice{Mode: ir.ToolChoiceAuto}
}

// ---- encode ----

func isReasoningModel(model string) bool {
	m := strings.ToLower(model)
	if i := strings.LastIndex(m, "/"); i >= 0 {
		m = m[i+1:]
	}
	for _, p := range []string{"o1", "o3", "o4", "gpt-5", "codex"} {
		if strings.HasPrefix(m, p) {
			return !strings.HasPrefix(m, "gpt-5-chat")
		}
	}
	return false
}

func (Codec) EncodeRequest(r *ir.Request) ([]byte, error) {
	out := obj{"model": r.Model, "store": false}

	var sys []string
	for _, p := range r.System {
		if p.Text != "" {
			sys = append(sys, p.Text)
		}
	}
	if len(sys) > 0 {
		out["instructions"] = strings.Join(sys, "\n\n")
	}

	custom := map[string]bool{}
	for _, t := range r.Tools {
		if t.Custom {
			custom[t.Name] = true
		}
	}
	customCalls := map[string]bool{}

	input := []obj{}
	for _, m := range ir.RepairToolPairing(r.Messages, &r.Diags) {
		if m.Role == ir.RoleAssistant {
			input = append(input, encodeAssistant(m.Parts, custom, customCalls, r)...)
		} else {
			input = append(input, encodeUser(m.Parts, customCalls, r)...)
		}
	}
	out["input"] = input

	if len(r.Tools) > 0 {
		tools := make([]obj, 0, len(r.Tools))
		for _, t := range r.Tools {
			if t.Custom {
				tools = append(tools, obj{"type": "custom", "name": t.Name, "description": t.Description})
				continue
			}
			strict := false
			if t.Strict != nil {
				strict = *t.Strict
			}
			tools = append(tools, obj{"type": "function", "name": t.Name, "description": t.Description,
				"parameters": ir.SchemaOrEmpty(t.Parameters), "strict": strict})
		}
		out["tools"] = tools
		if tc := r.ToolChoice; tc != nil {
			if tc.Mode == ir.ToolChoiceTool {
				typ := "function"
				if custom[tc.Name] {
					typ = "custom"
				}
				out["tool_choice"] = obj{"type": typ, "name": tc.Name}
			} else {
				out["tool_choice"] = string(tc.Mode)
			}
		}
		if r.ParallelToolCalls != nil {
			out["parallel_tool_calls"] = *r.ParallelToolCalls
		}
	}

	if r.MaxTokens != nil {
		out["max_output_tokens"] = max(*r.MaxTokens, 16)
	}
	if isReasoningModel(r.Model) {
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
	if len(r.Stop) > 0 {
		r.Info("stop_dropped", "stop sequences are not supported by the Responses API")
	}
	if r.TopK != nil {
		r.Info("top_k_dropped", "top_k is not supported by the Responses API")
	}
	if rs := r.Reasoning; rs != nil {
		if e := rs.EffortLevel(); e != "" {
			ro := obj{"effort": e}
			if e != "none" {
				summary := rs.Summary
				if summary == "" {
					summary = "auto"
				}
				ro["summary"] = summary
			}
			out["reasoning"] = ro
		}
	}
	out["include"] = []string{"reasoning.encrypted_content"}
	if rf := r.ResponseFormat; rf != nil && rf.Type != "" && rf.Type != "text" {
		f := obj{"type": rf.Type}
		if rf.Type == "json_schema" {
			f["name"] = rf.Name
			if rf.Name == "" {
				f["name"] = "response"
			}
			f["schema"] = rf.Schema
			if rf.Strict != nil {
				f["strict"] = *rf.Strict
			}
		}
		out["text"] = obj{"format": f}
	}
	if r.Stream {
		out["stream"] = true
	}
	if r.User != "" {
		out["user"] = r.User
	}
	return json.Marshal(out)
}

func contentParts(parts []ir.Part, r *ir.Request) []obj {
	var out []obj
	for _, p := range parts {
		switch p.Type {
		case ir.PartText:
			if p.Text != "" {
				out = append(out, obj{"type": "input_text", "text": p.Text})
			}
		case ir.PartImage:
			o := obj{"type": "input_image", "image_url": p.Media.URLOf()}
			if p.Media.Detail != "" {
				o["detail"] = p.Media.Detail
			} else {
				o["detail"] = "auto"
			}
			out = append(out, o)
		case ir.PartFile:
			o := obj{"type": "input_file"}
			if p.Media.Data != "" {
				o["file_data"] = p.Media.URLOf()
				o["filename"] = p.Media.Filename
				if p.Media.Filename == "" {
					o["filename"] = "file.pdf"
				}
			} else {
				o["file_url"] = p.Media.URL
			}
			out = append(out, o)
		}
	}
	return out
}

func encodeUser(parts []ir.Part, customCalls map[string]bool, r *ir.Request) []obj {
	var out []obj
	var rest []ir.Part
	for _, p := range parts {
		if p.Type != ir.PartToolResult {
			rest = append(rest, p)
			continue
		}
		typ := "function_call_output"
		if customCalls[p.ToolCallID] {
			typ = "custom_tool_call_output"
		}
		var output any
		if hasMedia(p.Content) && typ == "function_call_output" {
			output = contentParts(p.Content, r)
		} else {
			t := ir.TextOf(p.Content)
			if p.IsError {
				t = "Error: " + t
			}
			output = t
		}
		out = append(out, obj{"type": typ, "call_id": p.ToolCallID, "output": output})
	}
	if c := contentParts(rest, r); len(c) > 0 {
		out = append(out, obj{"type": "message", "role": "user", "content": c})
	}
	return out
}

func hasMedia(parts []ir.Part) bool {
	for _, p := range parts {
		if p.Type == ir.PartImage || p.Type == ir.PartFile {
			return true
		}
	}
	return false
}

func encodeAssistant(parts []ir.Part, custom, customCalls map[string]bool, r *ir.Request) []obj {
	var out []obj
	var text strings.Builder
	flush := func() {
		if text.Len() > 0 {
			out = append(out, obj{"type": "message", "role": "assistant",
				"content": []obj{{"type": "output_text", "text": text.String()}}})
			text.Reset()
		}
	}
	for _, p := range parts {
		switch p.Type {
		case ir.PartText:
			text.WriteString(p.Text)
		case ir.PartThinking:
			if p.SigOrigin != self || p.Signature == "" {
				continue // only our own encrypted reasoning can be replayed
			}
			flush()
			id, enc := unpackReasoning(p.Signature)
			item := obj{"type": "reasoning", "encrypted_content": enc, "summary": []obj{}}
			if id != "" {
				item["id"] = id
			}
			if p.Text != "" {
				item["summary"] = []obj{{"type": "summary_text", "text": p.Text}}
			}
			out = append(out, item)
		case ir.PartToolCall:
			flush()
			if custom[p.ToolName] {
				customCalls[p.ToolCallID] = true
				var a struct {
					Input string `json:"input"`
				}
				_ = json.Unmarshal([]byte(p.Arguments), &a)
				out = append(out, obj{"type": "custom_tool_call", "call_id": p.ToolCallID, "name": p.ToolName, "input": a.Input})
				continue
			}
			args, ok := ir.NormalizeArgs(p.Arguments)
			if !ok {
				r.Warn("invalid_tool_arguments", "tool call "+p.ToolCallID+" has non-object arguments")
			}
			out = append(out, obj{"type": "function_call", "call_id": p.ToolCallID, "name": p.ToolName, "arguments": args})
		}
	}
	flush()
	return out
}
