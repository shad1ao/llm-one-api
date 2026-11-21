package ir

import (
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"strings"
)

// ---- stop reasons: the single source of truth for mapping ----

func StopFromClaude(s string) StopReason {
	switch s {
	case "end_turn", "":
		return StopEndTurn
	case "max_tokens", "model_context_window_exceeded":
		return StopMaxTokens
	case "tool_use":
		return StopToolUse
	case "stop_sequence":
		return StopSequence
	case "refusal":
		return StopContentFilter
	case "pause_turn":
		return StopPause
	}
	return StopEndTurn
}

func StopToClaude(s StopReason) string {
	switch s {
	case StopMaxTokens:
		return "max_tokens"
	case StopToolUse:
		return "tool_use"
	case StopSequence:
		return "stop_sequence"
	case StopContentFilter:
		return "refusal"
	case StopPause:
		return "pause_turn"
	}
	return "end_turn"
}

func StopFromOpenAI(s string) StopReason {
	switch s {
	case "length":
		return StopMaxTokens
	case "tool_calls", "function_call":
		return StopToolUse
	case "content_filter":
		return StopContentFilter
	}
	return StopEndTurn
}

func StopToOpenAI(s StopReason) string {
	switch s {
	case StopMaxTokens, StopPause:
		return "length"
	case StopToolUse:
		return "tool_calls"
	case StopContentFilter:
		return "content_filter"
	}
	return "stop"
}

func StopFromGemini(s string) StopReason {
	switch s {
	case "MAX_TOKENS":
		return StopMaxTokens
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "LANGUAGE":
		return StopContentFilter
	case "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL":
		return StopError
	}
	return StopEndTurn
}

func StopToGemini(s StopReason) string {
	switch s {
	case StopMaxTokens, StopPause:
		return "MAX_TOKENS"
	case StopContentFilter:
		return "SAFETY"
	case StopError:
		return "OTHER"
	}
	return "STOP"
}

// ---- reasoning effort <-> budget ----

var effortBudget = map[string]int{
	"minimal": 1024,
	"low":     2048,
	"medium":  8192,
	"high":    16384,
	"xhigh":   32768,
	"max":     32768,
}

// EffortToBudget converts an effort level to a thinking token budget.
// Returns 0 for "none" or unknown.
func EffortToBudget(effort string) int { return effortBudget[strings.ToLower(effort)] }

// BudgetToEffort converts a budget to the closest effort level.
func BudgetToEffort(budget int) string {
	switch {
	case budget <= 0:
		return "medium"
	case budget <= 1024:
		return "minimal"
	case budget <= 4096:
		return "low"
	case budget <= 12288:
		return "medium"
	case budget <= 24576:
		return "high"
	}
	return "xhigh"
}

// Enabled reports whether reasoning is requested at all.
func (r *Reasoning) Enabled() bool {
	if r == nil {
		return false
	}
	if r.BudgetTokens > 0 {
		return true
	}
	return r.Effort != "" && r.Effort != "none"
}

// Budget returns an explicit budget, or one derived from effort.
func (r *Reasoning) Budget() int {
	if r == nil {
		return 0
	}
	if r.BudgetTokens > 0 {
		return r.BudgetTokens
	}
	return EffortToBudget(r.Effort)
}

// EffortLevel returns the effort, deriving it from the budget if needed.
func (r *Reasoning) EffortLevel() string {
	if r == nil {
		return ""
	}
	if r.Effort != "" {
		return r.Effort
	}
	if r.BudgetTokens > 0 {
		return BudgetToEffort(r.BudgetTokens)
	}
	return ""
}

// ---- errors ----

// ErrorTypeForStatus picks a neutral error type for an HTTP status.
func ErrorTypeForStatus(status int) string {
	switch {
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusForbidden:
		return "permission_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status == http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status == 529:
		return "overloaded_error"
	case status >= 500:
		return "api_error"
	}
	return "invalid_request_error"
}

// ---- ids ----

func NewID(prefix string) string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return prefix + hex.EncodeToString(b[:])
}
