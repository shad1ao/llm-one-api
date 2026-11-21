package ir

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"
)

// ---- signature envelopes ----
//
// Thinking signatures (Claude signature, Responses encrypted_content, Gemini
// thoughtSignature) are only valid for the provider that issued them. When a
// signature has to travel through a client of a different format, it is
// wrapped so that it can be recognised when the client replays it, and only
// forwarded to an upstream of the matching format.

const sigPrefix = "llm1:"

// WrapSig tags a signature with its origin format.
func WrapSig(origin Format, sig string) string { return sigPrefix + string(origin) + ":" + sig }

// UnwrapSig parses a possibly wrapped signature. Unwrapped signatures are
// attributed to native (the format the client speaks).
func UnwrapSig(s string, native Format) (Format, string) {
	if rest, ok := strings.CutPrefix(s, sigPrefix); ok {
		if f, sig, ok := strings.Cut(rest, ":"); ok && Format(f).Valid() {
			return Format(f), sig
		}
	}
	return native, s
}

// SigFor renders a signature for a client speaking target.
func SigFor(target, origin Format, sig string) string {
	if sig == "" {
		return ""
	}
	if origin == target || origin == "" {
		return sig
	}
	return WrapSig(origin, sig)
}

// ---- tool-call id packing ----
//
// Gemini attaches a thoughtSignature to function calls which must be echoed
// back on the next turn. Clients of other formats have no field for it, so the
// signature rides inside the tool call id (base64url, which is valid in every
// format's id alphabet).

const callSigSep = "__llm1"

var originCode = map[Format]string{FormatGemini: "g", FormatOpenAIResponses: "r", FormatClaude: "c", FormatOpenAIChat: "o"}

// PackCallID embeds sig into id. No-op when sig is empty or origin == target.
func PackCallID(target Format, id string, origin Format, sig string) string {
	if sig == "" || origin == target {
		return id
	}
	code, ok := originCode[origin]
	if !ok {
		return id
	}
	return id + callSigSep + code + "_" + base64.RawURLEncoding.EncodeToString([]byte(sig))
}

// UnpackCallID reverses PackCallID.
func UnpackCallID(id string) (base string, origin Format, sig string) {
	head, rest, found := strings.Cut(id, callSigSep)
	if !found || len(rest) < 2 || rest[1] != '_' {
		return id, "", ""
	}
	for f, c := range originCode {
		if c == rest[:1] {
			raw, err := base64.RawURLEncoding.DecodeString(rest[2:])
			if err != nil {
				return id, "", ""
			}
			return head, f, string(raw)
		}
	}
	return id, "", ""
}

// ---- media ----

// ParseDataURL splits "data:<mime>;base64,<data>". ok is false otherwise.
func ParseDataURL(u string) (mediaType, data string, ok bool) {
	rest, found := strings.CutPrefix(u, "data:")
	if !found {
		return "", "", false
	}
	meta, payload, found := strings.Cut(rest, ",")
	if !found {
		return "", "", false
	}
	mt, enc, _ := strings.Cut(meta, ";")
	if enc != "base64" {
		return "", "", false
	}
	return mt, payload, true
}

// DataURL builds a base64 data URL.
func DataURL(mediaType, data string) string { return "data:" + mediaType + ";base64," + data }

// MediaFromURL builds Media from a URL that may be a data URL.
func MediaFromURL(u string) *Media {
	if mt, data, ok := ParseDataURL(u); ok {
		return &Media{MediaType: mt, Data: data}
	}
	return &Media{URL: u}
}

// URLOf returns a URL for the media, building a data URL from inline data.
func (m *Media) URLOf() string {
	if m.URL != "" {
		return m.URL
	}
	mt := m.MediaType
	if mt == "" {
		mt = "application/octet-stream"
	}
	return DataURL(mt, m.Data)
}

// ---- JSON helpers ----

// NormalizeArgs returns compact JSON object text for tool arguments. Empty
// input becomes "{}". ok is false if s is not a JSON object.
func NormalizeArgs(s string) (string, bool) {
	s = strings.TrimSpace(s)
	if s == "" || s == "null" {
		return "{}", true
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, []byte(s)); err != nil {
		return s, false
	}
	out := buf.String()
	return out, strings.HasPrefix(out, "{")
}

// ArgsObject returns tool arguments as a JSON object, replacing invalid input
// with {} and recording a diagnostic on d (if non-nil).
func ArgsObject(s string, d *Diags) json.RawMessage {
	out, ok := NormalizeArgs(s)
	if !ok {
		if d != nil {
			d.Add(SevWarn, "invalid_tool_arguments", "tool arguments are not a JSON object; replaced with {}")
		}
		return json.RawMessage("{}")
	}
	return json.RawMessage(out)
}

// RawString renders a JSON value as tool-argument text.
func RawString(v json.RawMessage) string {
	if len(v) == 0 {
		return "{}"
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, v); err != nil {
		return string(v)
	}
	return buf.String()
}

// EmptySchema is the schema used when a tool declares no parameters.
var EmptySchema = json.RawMessage(`{"type":"object","properties":{}}`)

// SchemaOrEmpty returns s, or EmptySchema if s is empty/null.
func SchemaOrEmpty(s json.RawMessage) json.RawMessage {
	t := bytes.TrimSpace(s)
	if len(t) == 0 || bytes.Equal(t, []byte("null")) || bytes.Equal(t, []byte("{}")) {
		return EmptySchema
	}
	return s
}

// TextOrParts decodes a JSON value that is either a string or something else;
// it returns the string and true when it is a string.
func TextOrParts(raw json.RawMessage) (string, bool) {
	t := bytes.TrimSpace(raw)
	if len(t) > 0 && t[0] == '"' {
		var s string
		if json.Unmarshal(t, &s) == nil {
			return s, true
		}
	}
	return "", false
}

// MergeAdjacent merges consecutive messages with the same role.
func MergeAdjacent(msgs []Message) []Message {
	var out []Message
	for _, m := range msgs {
		if len(m.Parts) == 0 {
			continue
		}
		if n := len(out); n > 0 && out[n-1].Role == m.Role {
			out[n-1].Parts = append(out[n-1].Parts, m.Parts...)
			continue
		}
		out = append(out, Message{Role: m.Role, Parts: append([]Part(nil), m.Parts...)})
	}
	return out
}
