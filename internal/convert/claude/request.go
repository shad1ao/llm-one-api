// Package claude implements the Anthropic Messages API codec.
package claude

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
)

const self = ir.FormatClaude

// DefaultMaxTokens is used when the source request has no max_tokens (Claude
// requires the field).
var DefaultMaxTokens = 8192

func init() { convert.Register(Codec{}) }

// Codec is the Claude Messages codec.
type Codec struct{}

func (Codec) Format() ir.Format { return self }

// ---- wire types (decode side) ----

type wireRequest struct {
	Model         string          `json:"model"`
	MaxTokens     *int            `json:"max_tokens"`
	System        json.RawMessage `json:"system"`
	Messages      []wireMessage   `json:"messages"`
	Tools         []wireTool      `json:"tools"`
	ToolChoice    *wireToolChoice `json:"tool_choice"`
	Temperature   *float64        `json:"temperature"`
	TopP          *float64        `json:"top_p"`
	TopK          *int            `json:"top_k"`
	StopSequences []string        `json:"stop_sequences"`
	Stream        bool            `json:"stream"`
	Thinking      *struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens"`
	} `json:"thinking"`
	OutputConfig *struct {
		Effort string `json:"effort"`
	} `json:"output_config"`
	Metadata *struct {
		UserID string `json:"user_id"`
	} `json:"metadata"`
}

type wireMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type wireSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	URL       string `json:"url"`
}

type wireBlock struct {
	Type         string           `json:"type"`
	Text         string           `json:"text"`
	Source       *wireSource      `json:"source"`
	ID           string           `json:"id"`
	Name         string           `json:"name"`
	Input        json.RawMessage  `json:"input"`
	ToolUseID    string           `json:"tool_use_id"`
	Content      json.RawMessage  `json:"content"`
	IsError      bool             `json:"is_error"`
	Thinking     string           `json:"thinking"`
	Signature    string           `json:"signature"`
	Data         string           `json:"data"`
	Title        string           `json:"title"`
	CacheControl *ir.CacheControl `json:"cache_control"`
}

type wireTool struct {
	Type         string           `json:"type"`
	Name         string           `json:"name"`
	Description  string           `json:"description"`
	InputSchema  json.RawMessage  `json:"input_schema"`
	CacheControl *ir.CacheControl `json:"cache_control"`
}

type wireToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use"`
}

// ---- decode ----

func (Codec) DecodeRequest(body []byte) (*ir.Request, error) {
	var w wireRequest
	if err := json.Unmarshal(body, &w); err != nil {
		return nil, fmt.Errorf("invalid claude request: %w", err)
	}
	r := &ir.Request{
		Model:       w.Model,
		MaxTokens:   w.MaxTokens,
		Temperature: w.Temperature,
		TopP:        w.TopP,
		TopK:        w.TopK,
		Stop:        w.StopSequences,
		Stream:      w.Stream,
	}
	if w.Metadata != nil {
		r.User = w.Metadata.UserID
	}

	if s, ok := ir.TextOrParts(w.System); ok {
		if s != "" {
			r.System = []ir.Part{{Type: ir.PartText, Text: s}}
		}
	} else if len(w.System) > 0 && string(w.System) != "null" {
		var blocks []wireBlock
		if err := json.Unmarshal(w.System, &blocks); err != nil {
			return nil, fmt.Errorf("invalid claude system: %w", err)
		}
		for _, b := range blocks {
			if b.Type == "text" {
				r.System = append(r.System, ir.Part{Type: ir.PartText, Text: b.Text, Cache: b.CacheControl})
			}
		}
	}

	for i, m := range w.Messages {
		parts, err := decodeContent(m.Content, r)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %w", i, err)
		}
		role := ir.RoleUser
		if m.Role == "assistant" {
			role = ir.RoleAssistant
		}
		r.Messages = append(r.Messages, ir.Message{Role: role, Parts: parts})
	}

	for _, t := range w.Tools {
		if t.Type != "" && t.Type != "custom" {
			r.Warn("hosted_tool_dropped", "claude server tool "+t.Type+" ("+t.Name+") is not portable")
			continue
		}
		r.Tools = append(r.Tools, ir.Tool{Name: t.Name, Description: t.Description,
			Parameters: ir.SchemaOrEmpty(t.InputSchema), Cache: t.CacheControl})
	}

	if tc := w.ToolChoice; tc != nil {
		switch tc.Type {
		case "auto":
			r.ToolChoice = &ir.ToolChoice{Mode: ir.ToolChoiceAuto}
		case "any":
			r.ToolChoice = &ir.ToolChoice{Mode: ir.ToolChoiceRequired}
		case "none":
			r.ToolChoice = &ir.ToolChoice{Mode: ir.ToolChoiceNone}
		case "tool":
			r.ToolChoice = &ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: tc.Name}
		}
		if tc.DisableParallelToolUse {
			f := false
			r.ParallelToolCalls = &f
		}
	}

	if th := w.Thinking; th != nil {
		switch th.Type {
		case "enabled":
			r.Reasoning = &ir.Reasoning{BudgetTokens: th.BudgetTokens}
		case "adaptive":
			r.Reasoning = &ir.Reasoning{Effort: "medium"}
		case "disabled":
			r.Reasoning = &ir.Reasoning{Effort: "none"}
		}
	}
	if oc := w.OutputConfig; oc != nil && oc.Effort != "" && (r.Reasoning == nil || r.Reasoning.Effort != "none") {
		if r.Reasoning == nil {
			r.Reasoning = &ir.Reasoning{}
		}
		e := oc.Effort
		if e == "max" {
			e = "xhigh"
		}
		r.Reasoning.Effort = e
	}
	return r, nil
}

func decodeContent(raw json.RawMessage, r *ir.Request) ([]ir.Part, error) {
	if s, ok := ir.TextOrParts(raw); ok {
		if s == "" {
			return nil, nil
		}
		return []ir.Part{{Type: ir.PartText, Text: s}}, nil
	}
	var blocks []wireBlock
	if len(raw) > 0 && string(raw) != "null" {
		if err := json.Unmarshal(raw, &blocks); err != nil {
			return nil, err
		}
	}
	var parts []ir.Part
	for _, b := range blocks {
		switch b.Type {
		case "text":
			parts = append(parts, ir.Part{Type: ir.PartText, Text: b.Text, Cache: b.CacheControl})
		case "image", "document":
			if p, ok := decodeMedia(b, r); ok {
				parts = append(parts, p)
			}
		case "tool_use":
			id, origin, sig := ir.UnpackCallID(b.ID)
			parts = append(parts, ir.Part{Type: ir.PartToolCall, ToolCallID: id, ToolName: b.Name,
				Arguments: ir.RawString(b.Input), Signature: sig, SigOrigin: origin, Cache: b.CacheControl})
		case "tool_result":
			id, _, _ := ir.UnpackCallID(b.ToolUseID)
			p := ir.Part{Type: ir.PartToolResult, ToolCallID: id, IsError: b.IsError, Cache: b.CacheControl}
			if s, ok := ir.TextOrParts(b.Content); ok {
				p.Content = []ir.Part{{Type: ir.PartText, Text: s}}
			} else if len(b.Content) > 0 && string(b.Content) != "null" {
				inner, err := decodeContent(b.Content, r)
				if err != nil {
					return nil, err
				}
				p.Content = inner
			}
			parts = append(parts, p)
		case "thinking":
			origin, sig := ir.UnwrapSig(b.Signature, self)
			parts = append(parts, ir.Part{Type: ir.PartThinking, Text: b.Thinking, Signature: sig, SigOrigin: origin})
		case "redacted_thinking":
			parts = append(parts, ir.Part{Type: ir.PartRedactedThinking, Signature: b.Data, SigOrigin: self})
		case "server_tool_use", "web_search_tool_result", "web_fetch_tool_result", "code_execution_tool_result",
			"mcp_tool_use", "mcp_tool_result", "container_upload":
			r.Info("hosted_block_dropped", "claude "+b.Type+" block dropped")
		default:
			r.Warn("unknown_block", "unknown claude content block "+b.Type)
		}
	}
	return parts, nil
}

func decodeMedia(b wireBlock, r *ir.Request) (ir.Part, bool) {
	typ := ir.PartImage
	if b.Type == "document" {
		typ = ir.PartFile
	}
	if b.Source == nil {
		return ir.Part{}, false
	}
	m := &ir.Media{MediaType: b.Source.MediaType, Filename: b.Title}
	switch b.Source.Type {
	case "base64":
		m.Data = b.Source.Data
	case "url":
		m.URL = b.Source.URL
	case "text":
		return ir.Part{Type: ir.PartText, Text: b.Source.Data, Cache: b.CacheControl}, true
	default:
		r.Warn("unsupported_media_source", "claude media source "+b.Source.Type+" dropped")
		return ir.Part{}, false
	}
	return ir.Part{Type: typ, Media: m, Cache: b.CacheControl}, true
}

// ---- encode ----

type obj = map[string]any

var validID = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// toolID makes an id acceptable to Claude (^[a-zA-Z0-9_-]+$).
func toolID(id string) string {
	if id == "" {
		return ir.NewID("toolu_")
	}
	return validID.ReplaceAllString(id, "_")
}

func (Codec) EncodeRequest(r *ir.Request) ([]byte, error) {
	out := obj{"model": r.Model}

	maxTokens := DefaultMaxTokens
	if r.MaxTokens != nil && *r.MaxTokens > 0 {
		maxTokens = *r.MaxTokens
	}

	msgs := ir.RepairToolPairing(r.Messages, &r.Diags)
	thinking := r.Reasoning.Enabled()
	if thinking && lastToolTurnLacksThinking(msgs) {
		r.Info("thinking_disabled", "disabled thinking: the in-progress tool turn has no Claude-signed thinking block")
		thinking = false
	}

	system := make([]obj, 0, len(r.System))
	for _, p := range r.System {
		if p.Text == "" {
			continue
		}
		system = append(system, withCache(obj{"type": "text", "text": p.Text}, p.Cache))
	}
	if rf := r.ResponseFormat; rf != nil && rf.Type != "text" && rf.Type != "" {
		hint := "Respond only with a single valid JSON object and no other text."
		if len(rf.Schema) > 0 {
			hint += " The JSON must conform to this JSON schema:\n" + string(rf.Schema)
		}
		system = append(system, obj{"type": "text", "text": hint})
		r.Info("response_format_emulated", "response_format emulated with a system instruction")
	}
	if len(system) > 0 {
		out["system"] = system
	}

	wmsgs := make([]obj, 0, len(msgs))
	for _, m := range msgs {
		blocks := encodeParts(m.Parts, m.Role, r)
		if len(blocks) == 0 {
			continue
		}
		role := "user"
		if m.Role == ir.RoleAssistant {
			role = "assistant"
		}
		wmsgs = append(wmsgs, obj{"role": role, "content": blocks})
	}
	out["messages"] = wmsgs

	if len(r.Tools) > 0 {
		tools := make([]obj, 0, len(r.Tools))
		for _, t := range r.Tools {
			tools = append(tools, withCache(obj{"name": t.Name, "description": t.Description,
				"input_schema": ir.SchemaOrEmpty(t.Parameters)}, t.Cache))
		}
		out["tools"] = tools
	}

	if len(r.Tools) > 0 {
		var tc obj
		if r.ToolChoice != nil {
			switch r.ToolChoice.Mode {
			case ir.ToolChoiceAuto:
				tc = obj{"type": "auto"}
			case ir.ToolChoiceNone:
				tc = obj{"type": "none"}
			case ir.ToolChoiceRequired:
				tc = obj{"type": "any"}
			case ir.ToolChoiceTool:
				tc = obj{"type": "tool", "name": r.ToolChoice.Name}
			}
			if thinking && (tc["type"] == "any" || tc["type"] == "tool") {
				r.Info("tool_choice_relaxed", "forced tool_choice is incompatible with thinking; using auto")
				tc = obj{"type": "auto"}
			}
		}
		if r.ParallelToolCalls != nil && !*r.ParallelToolCalls {
			if tc == nil {
				tc = obj{"type": "auto"}
			}
			if tc["type"] != "none" {
				tc["disable_parallel_tool_use"] = true
			}
		}
		if tc != nil {
			out["tool_choice"] = tc
		}
	}

	if thinking {
		budget := max(r.Reasoning.Budget(), 1024)
		if maxTokens <= budget {
			maxTokens = budget + maxTokens
		}
		out["thinking"] = obj{"type": "enabled", "budget_tokens": budget}
		if r.TopP != nil && *r.TopP >= 0.95 {
			out["top_p"] = *r.TopP
		}
	} else {
		if r.Temperature != nil {
			out["temperature"] = min(*r.Temperature, 1.0)
		}
		if r.TopP != nil {
			out["top_p"] = *r.TopP
		}
		if r.TopK != nil {
			out["top_k"] = *r.TopK
		}
	}
	out["max_tokens"] = maxTokens

	var stops []string
	for _, s := range r.Stop {
		if strings.TrimSpace(s) != "" {
			stops = append(stops, s)
		}
	}
	if len(stops) > 0 {
		out["stop_sequences"] = stops
	}
	if r.Stream {
		out["stream"] = true
	}
	if r.User != "" {
		out["metadata"] = obj{"user_id": r.User}
	}
	return json.Marshal(out)
}

func withCache(o obj, c *ir.CacheControl) obj {
	if c != nil {
		o["cache_control"] = c
	}
	return o
}

// lastToolTurnLacksThinking reports whether the conversation ends in an
// in-progress tool loop whose assistant turn has no replayable thinking.
// Claude rejects such requests when thinking is enabled.
func lastToolTurnLacksThinking(msgs []ir.Message) bool {
	if len(msgs) < 2 || msgs[len(msgs)-1].Role != ir.RoleUser {
		return false
	}
	last := msgs[len(msgs)-1]
	hasResult := false
	for _, p := range last.Parts {
		if p.Type == ir.PartToolResult {
			hasResult = true
		}
	}
	if !hasResult {
		return false
	}
	asst := msgs[len(msgs)-2]
	for _, p := range asst.Parts {
		if (p.Type == ir.PartThinking || p.Type == ir.PartRedactedThinking) && p.SigOrigin == self && p.Signature != "" {
			return false
		}
	}
	return true
}

func encodeParts(parts []ir.Part, role ir.Role, r *ir.Request) []obj {
	blocks := make([]obj, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case ir.PartText:
			if p.Text == "" {
				continue
			}
			blocks = append(blocks, withCache(obj{"type": "text", "text": p.Text}, p.Cache))
		case ir.PartImage, ir.PartFile:
			if b := encodeMedia(p, r); b != nil {
				blocks = append(blocks, b)
			}
		case ir.PartToolCall:
			b := obj{"type": "tool_use", "id": toolID(p.ToolCallID), "name": p.ToolName,
				"input": ir.ArgsObject(p.Arguments, &r.Diags)}
			blocks = append(blocks, withCache(b, p.Cache))
		case ir.PartToolResult:
			b := obj{"type": "tool_result", "tool_use_id": toolID(p.ToolCallID)}
			if p.IsError {
				b["is_error"] = true
			}
			if hasMedia(p.Content) {
				b["content"] = encodeParts(p.Content, role, r)
			} else if t := ir.TextOf(p.Content); t != "" {
				b["content"] = t
			}
			blocks = append(blocks, withCache(b, p.Cache))
		case ir.PartThinking:
			if role != ir.RoleAssistant || p.SigOrigin != self || p.Signature == "" {
				continue // unsigned or foreign thinking cannot be replayed
			}
			blocks = append(blocks, obj{"type": "thinking", "thinking": p.Text, "signature": p.Signature})
		case ir.PartRedactedThinking:
			if role == ir.RoleAssistant && p.SigOrigin == self {
				blocks = append(blocks, obj{"type": "redacted_thinking", "data": p.Signature})
			}
		}
	}
	return blocks
}

func hasMedia(parts []ir.Part) bool {
	for _, p := range parts {
		if p.Type == ir.PartImage || p.Type == ir.PartFile {
			return true
		}
	}
	return false
}

func encodeMedia(p ir.Part, r *ir.Request) obj {
	m := p.Media
	if m == nil {
		return nil
	}
	typ := "image"
	if p.Type == ir.PartFile {
		typ = "document"
	}
	var src obj
	if m.Data != "" {
		src = obj{"type": "base64", "media_type": m.MediaType, "data": m.Data}
	} else if mt, data, ok := ir.ParseDataURL(m.URL); ok {
		src = obj{"type": "base64", "media_type": mt, "data": data}
	} else if m.URL != "" {
		src = obj{"type": "url", "url": m.URL}
	} else {
		r.Warn("unsupported_media_source", "media part without data or url dropped")
		return nil
	}
	if typ == "document" && src["type"] == "base64" && src["media_type"] != "application/pdf" {
		mt, _ := src["media_type"].(string)
		if strings.HasPrefix(mt, "text/") {
			raw, err := base64.StdEncoding.DecodeString(src["data"].(string))
			if err != nil {
				r.Warn("unsupported_media_source", "invalid base64 text document dropped")
				return nil
			}
			src = obj{"type": "text", "media_type": "text/plain", "data": string(raw)}
			b := obj{"type": typ, "source": src}
			if m.Filename != "" {
				b["title"] = m.Filename
			}
			return withCache(b, p.Cache)
		}
		if mt != "" {
			r.Warn("unsupported_media_type", "claude documents support PDF only; got "+mt)
			return nil
		}
		src["media_type"] = "application/pdf"
	}
	b := obj{"type": typ, "source": src}
	if typ == "document" && m.Filename != "" {
		b["title"] = m.Filename
	}
	return withCache(b, p.Cache)
}
