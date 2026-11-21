// Package ir defines the neutral intermediate representation that every wire
// format (OpenAI Chat, OpenAI Responses, Claude Messages, Gemini) is decoded
// into and encoded from. Conversion is always wire A -> IR -> wire B.
//
// Conventions:
//   - System prompts live in Request.System, never in Messages.
//   - Tool results are parts of a RoleUser message (Claude/Gemini style); the
//     OpenAI encoders split them into separate tool messages / items.
//   - Usage.Input excludes cache reads/writes (Claude semantics).
package ir

import (
	"encoding/json"
	"strings"
)

// Format identifies a wire protocol.
type Format string

const (
	FormatOpenAIChat      Format = "openai_chat"
	FormatOpenAIResponses Format = "openai_responses"
	FormatClaude          Format = "claude"
	FormatGemini          Format = "gemini"
)

// Valid reports whether f is a known format.
func (f Format) Valid() bool {
	switch f {
	case FormatOpenAIChat, FormatOpenAIResponses, FormatClaude, FormatGemini:
		return true
	}
	return false
}

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type PartType string

const (
	PartText             PartType = "text"
	PartImage            PartType = "image"
	PartFile             PartType = "file"
	PartToolCall         PartType = "tool_call"
	PartToolResult       PartType = "tool_result"
	PartThinking         PartType = "thinking"
	PartRedactedThinking PartType = "redacted_thinking"
)

// CacheControl mirrors Claude's prompt-caching breakpoint marker.
type CacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// Media is an image or file payload. Exactly one of Data (base64, no data:
// prefix) or URL is normally set.
type Media struct {
	MediaType string
	Data      string
	URL       string
	Filename  string
	Detail    string // OpenAI image detail: low|high|auto
}

// Part is one content element of a message.
type Part struct {
	Type PartType

	// PartText / PartThinking
	Text string

	// PartImage / PartFile
	Media *Media

	// PartToolCall
	ToolCallID string
	ToolName   string
	Arguments  string // raw JSON object text

	// PartToolResult (ToolCallID references the call)
	Content []Part // text / image parts
	IsError bool

	// PartThinking signature, PartRedactedThinking data, or Gemini
	// thoughtSignature attached to a tool call. SigOrigin is the format that
	// produced the signature; signatures are only replayed to that format.
	Signature string
	SigOrigin Format

	Cache *CacheControl
}

// TextOf concatenates all text parts.
func TextOf(parts []Part) string {
	var b strings.Builder
	for _, p := range parts {
		if p.Type == PartText {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

type Message struct {
	Role  Role
	Parts []Part
}

type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage // JSON schema object
	Strict      *bool
	// Custom marks an OpenAI Responses "custom" (freeform) tool which we
	// expose to other formats as a function with a single string "input".
	Custom bool
	Cache  *CacheControl
}

type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceTool     ToolChoiceMode = "tool"
)

type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}

// Reasoning describes the requested thinking behaviour. Effort is one of
// none|minimal|low|medium|high|xhigh. BudgetTokens > 0 means an explicit
// budget was requested.
type Reasoning struct {
	Effort       string
	BudgetTokens int
	Summary      string
}

type ResponseFormat struct {
	Type   string // text | json_object | json_schema
	Name   string
	Schema json.RawMessage
	Strict *bool
}

type Request struct {
	Model    string
	System   []Part
	Messages []Message

	Tools             []Tool
	ToolChoice        *ToolChoice
	ParallelToolCalls *bool

	MaxTokens   *int
	Temperature *float64
	TopP        *float64
	TopK        *int
	Stop        []string
	Seed        *int64

	Stream       bool
	IncludeUsage bool

	Reasoning      *Reasoning
	ResponseFormat *ResponseFormat

	User string

	// Diags collects non-fatal conversion losses.
	Diags Diags
}

// Warn records a conversion diagnostic.
func (r *Request) Warn(code, msg string) { r.Diags.Add(SevWarn, code, msg) }

// Info records an informational diagnostic.
func (r *Request) Info(code, msg string) { r.Diags.Add(SevInfo, code, msg) }

// ToolByName returns the tool declaration with the given name.
func (r *Request) ToolByName(name string) *Tool {
	if r == nil {
		return nil
	}
	for i := range r.Tools {
		if r.Tools[i].Name == name {
			return &r.Tools[i]
		}
	}
	return nil
}

type StopReason string

const (
	StopEndTurn       StopReason = "end_turn"
	StopMaxTokens     StopReason = "max_tokens"
	StopToolUse       StopReason = "tool_use"
	StopSequence      StopReason = "stop_sequence"
	StopContentFilter StopReason = "content_filter"
	StopPause         StopReason = "pause_turn"
	StopError         StopReason = "error"
)

// Usage token counts. Input excludes CacheRead and CacheWrite. Output
// includes Reasoning. CacheWrite1h is the 1-hour TTL portion of CacheWrite.
type Usage struct {
	Input        int
	Output       int
	CacheRead    int
	CacheWrite   int
	CacheWrite1h int
	Reasoning    int
}

// TotalInput returns input including cached tokens (OpenAI semantics).
func (u Usage) TotalInput() int { return u.Input + u.CacheRead + u.CacheWrite }

// Merge overwrites fields with positive values from o; zeros never erase.
func (u *Usage) Merge(o Usage) {
	set := func(dst *int, v int) {
		if v > 0 {
			*dst = v
		}
	}
	set(&u.Input, o.Input)
	set(&u.Output, o.Output)
	set(&u.CacheRead, o.CacheRead)
	set(&u.CacheWrite, o.CacheWrite)
	set(&u.CacheWrite1h, o.CacheWrite1h)
	set(&u.Reasoning, o.Reasoning)
}

// UsageFromTotal builds Usage from OpenAI/Gemini style totals where
// promptTotal already includes cached tokens.
func UsageFromTotal(promptTotal, cached, output, reasoning int) Usage {
	in := max(promptTotal-cached, 0)
	return Usage{Input: in, CacheRead: cached, Output: output, Reasoning: reasoning}
}

type Response struct {
	ID         string
	Model      string
	Parts      []Part
	StopReason StopReason
	StopSeq    string
	Usage      Usage
}

// HasToolCalls reports whether the response contains any tool call.
func (r *Response) HasToolCalls() bool {
	for _, p := range r.Parts {
		if p.Type == PartToolCall {
			return true
		}
	}
	return false
}

// APIError is a provider-neutral error.
type APIError struct {
	Status  int
	Type    string
	Message string
}

func (e *APIError) Error() string { return e.Message }
