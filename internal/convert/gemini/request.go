// Package gemini implements the Google Gemini generateContent codec.
//
// The model name and streaming flag live in the URL for Gemini, not in the
// body: callers set ir.Request.Model / Stream from the path when decoding and
// build the URL themselves when encoding.
package gemini

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
)

const self = ir.FormatGemini

// SkipSignature is Google's documented placeholder for function calls whose
// thought signature is unknown (e.g. history produced by another model).
const SkipSignature = "skip_thought_signature_validator"

func init() { convert.Register(Codec{}) }

// Codec is the Gemini codec.
type Codec struct{}

func (Codec) Format() ir.Format { return self }

type obj = map[string]any

// ---- signatures ----
//
// thoughtSignature is a base64 bytes field. Foreign signatures are wrapped and
// base64-encoded so they stay valid base64 for Gemini SDKs.

func sigToClient(origin ir.Format, sig string) string {
	if sig == "" {
		return ""
	}
	if origin == self || origin == "" {
		return sig
	}
	return base64.StdEncoding.EncodeToString([]byte(ir.WrapSig(origin, sig)))
}

func sigFromClient(s string) (ir.Format, string) {
	if s == "" {
		return "", ""
	}
	if raw, err := base64.StdEncoding.DecodeString(s); err == nil && strings.HasPrefix(string(raw), "llm1:") {
		return ir.UnwrapSig(string(raw), self)
	}
	return self, s
}

// ---- wire types ----

type wireFunctionCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type wireFunctionResponse struct {
	ID       string          `json:"id"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type wirePart struct {
	Text             *string               `json:"text"`
	Thought          bool                  `json:"thought"`
	ThoughtSignature string                `json:"thoughtSignature"`
	InlineData       *wireBlob             `json:"inlineData"`
	FileData         *wireFile             `json:"fileData"`
	FunctionCall     *wireFunctionCall     `json:"functionCall"`
	FunctionResponse *wireFunctionResponse `json:"functionResponse"`
}

type wireBlob struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type wireFile struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type wireContent struct {
	Role  string     `json:"role"`
	Parts []wirePart `json:"parts"`
}

type wireRequest struct {
	Contents          []wireContent `json:"contents"`
	SystemInstruction *wireContent  `json:"systemInstruction"`
	Tools             []struct {
		FunctionDeclarations []struct {
			Name                 string          `json:"name"`
			Description          string          `json:"description"`
			Parameters           json.RawMessage `json:"parameters"`
			ParametersJSONSchema json.RawMessage `json:"parametersJsonSchema"`
		} `json:"functionDeclarations"`
		GoogleSearch  json.RawMessage `json:"googleSearch"`
		CodeExecution json.RawMessage `json:"codeExecution"`
	} `json:"tools"`
	ToolConfig *struct {
		FunctionCallingConfig *struct {
			Mode                 string   `json:"mode"`
			AllowedFunctionNames []string `json:"allowedFunctionNames"`
		} `json:"functionCallingConfig"`
	} `json:"toolConfig"`
	GenerationConfig *struct {
		Temperature        *float64        `json:"temperature"`
		TopP               *float64        `json:"topP"`
		TopK               *float64        `json:"topK"`
		MaxOutputTokens    *int            `json:"maxOutputTokens"`
		StopSequences      []string        `json:"stopSequences"`
		Seed               *int64          `json:"seed"`
		ResponseMimeType   string          `json:"responseMimeType"`
		ResponseSchema     json.RawMessage `json:"responseSchema"`
		ResponseJSONSchema json.RawMessage `json:"responseJsonSchema"`
		ThinkingConfig     *struct {
			ThinkingBudget  *int   `json:"thinkingBudget"`
			ThinkingLevel   string `json:"thinkingLevel"`
			IncludeThoughts bool   `json:"includeThoughts"`
		} `json:"thinkingConfig"`
	} `json:"generationConfig"`
}

// ---- decode ----

func (Codec) DecodeRequest(body []byte) (*ir.Request, error) {
	norm, err := normalizeBody(body)
	if err != nil {
		return nil, fmt.Errorf("invalid gemini request: %w", err)
	}
	var w wireRequest
	if err := json.Unmarshal(norm, &w); err != nil {
		return nil, fmt.Errorf("invalid gemini request: %w", err)
	}
	r := &ir.Request{IncludeUsage: true}
	if si := w.SystemInstruction; si != nil {
		for _, p := range si.Parts {
			if p.Text != nil && *p.Text != "" {
				r.System = append(r.System, ir.Part{Type: ir.PartText, Text: *p.Text})
			}
		}
	}

	// Gemini pairs function responses with calls by name/order; assign ids.
	pending := map[string][]string{}
	n := 0
	for _, c := range w.Contents {
		role := ir.RoleUser
		if c.Role == "model" {
			role = ir.RoleAssistant
		}
		var parts []ir.Part
		for _, p := range c.Parts {
			switch {
			case p.FunctionCall != nil:
				fc := p.FunctionCall
				id, origin, sig := ir.UnpackCallID(fc.ID)
				if id == "" {
					n++
					id = fmt.Sprintf("call_%s_%d", sanitize(fc.Name), n)
				}
				if p.ThoughtSignature != "" && p.ThoughtSignature != SkipSignature {
					origin, sig = sigFromClient(p.ThoughtSignature)
				}
				pending[fc.Name] = append(pending[fc.Name], id)
				parts = append(parts, ir.Part{Type: ir.PartToolCall, ToolCallID: id, ToolName: fc.Name,
					Arguments: ir.RawString(fc.Args), Signature: sig, SigOrigin: origin})
			case p.FunctionResponse != nil:
				fr := p.FunctionResponse
				id, _, _ := ir.UnpackCallID(fr.ID)
				if q := pending[fr.Name]; id == "" && len(q) > 0 {
					id, pending[fr.Name] = q[0], q[1:]
				} else if id != "" {
					pending[fr.Name] = remove(q, id)
				}
				if id == "" {
					n++
					id = fmt.Sprintf("call_%s_%d", sanitize(fr.Name), n)
				}
				parts = append(parts, ir.Part{Type: ir.PartToolResult, ToolCallID: id, ToolName: fr.Name,
					Content: []ir.Part{{Type: ir.PartText, Text: responseText(fr.Response)}}})
			case p.Text != nil && p.Thought:
				origin, sig := sigFromClient(p.ThoughtSignature)
				parts = append(parts, ir.Part{Type: ir.PartThinking, Text: *p.Text, Signature: sig, SigOrigin: origin})
			case p.Text != nil:
				if *p.Text != "" {
					parts = append(parts, ir.Part{Type: ir.PartText, Text: *p.Text})
				}
			case p.InlineData != nil:
				parts = append(parts, mediaPart(p.InlineData.MimeType, &ir.Media{MediaType: p.InlineData.MimeType, Data: p.InlineData.Data}))
			case p.FileData != nil:
				parts = append(parts, mediaPart(p.FileData.MimeType, &ir.Media{MediaType: p.FileData.MimeType, URL: p.FileData.FileURI}))
			}
		}
		r.Messages = append(r.Messages, ir.Message{Role: role, Parts: parts})
	}
	r.Messages = ir.MergeAdjacent(r.Messages)

	for _, t := range w.Tools {
		for _, fd := range t.FunctionDeclarations {
			params := fd.ParametersJSONSchema
			if len(params) == 0 {
				params = lowerTypes(fd.Parameters)
			}
			r.Tools = append(r.Tools, ir.Tool{Name: fd.Name, Description: fd.Description, Parameters: ir.SchemaOrEmpty(params)})
		}
		if len(t.GoogleSearch) > 0 || len(t.CodeExecution) > 0 {
			r.Warn("hosted_tool_dropped", "gemini built-in tools are not portable")
		}
	}

	if tc := w.ToolConfig; tc != nil && tc.FunctionCallingConfig != nil {
		fc := tc.FunctionCallingConfig
		switch strings.ToUpper(fc.Mode) {
		case "AUTO":
			r.ToolChoice = &ir.ToolChoice{Mode: ir.ToolChoiceAuto}
		case "NONE":
			r.ToolChoice = &ir.ToolChoice{Mode: ir.ToolChoiceNone}
		case "ANY", "VALIDATED":
			if len(fc.AllowedFunctionNames) == 1 {
				r.ToolChoice = &ir.ToolChoice{Mode: ir.ToolChoiceTool, Name: fc.AllowedFunctionNames[0]}
			} else {
				r.ToolChoice = &ir.ToolChoice{Mode: ir.ToolChoiceRequired}
			}
		}
	}

	if g := w.GenerationConfig; g != nil {
		r.Temperature, r.TopP, r.MaxTokens, r.Stop, r.Seed = g.Temperature, g.TopP, g.MaxOutputTokens, g.StopSequences, g.Seed
		if g.TopK != nil {
			k := int(*g.TopK)
			r.TopK = &k
		}
		if g.ResponseMimeType == "application/json" {
			rf := &ir.ResponseFormat{Type: "json_object"}
			if len(g.ResponseJSONSchema) > 0 {
				rf = &ir.ResponseFormat{Type: "json_schema", Schema: g.ResponseJSONSchema}
			} else if len(g.ResponseSchema) > 0 {
				rf = &ir.ResponseFormat{Type: "json_schema", Schema: lowerTypes(g.ResponseSchema)}
			}
			r.ResponseFormat = rf
		}
		if tc := g.ThinkingConfig; tc != nil {
			switch {
			case tc.ThinkingLevel != "":
				r.Reasoning = &ir.Reasoning{Effort: strings.ToLower(tc.ThinkingLevel)}
			case tc.ThinkingBudget != nil && *tc.ThinkingBudget == 0:
				r.Reasoning = &ir.Reasoning{Effort: "none"}
			case tc.ThinkingBudget != nil && *tc.ThinkingBudget > 0:
				r.Reasoning = &ir.Reasoning{BudgetTokens: *tc.ThinkingBudget}
			case tc.ThinkingBudget != nil: // -1 = dynamic
				r.Reasoning = &ir.Reasoning{Effort: "medium"}
			}
		}
	}
	return r, nil
}

func mediaPart(mime string, m *ir.Media) ir.Part {
	if strings.HasPrefix(mime, "image/") {
		return ir.Part{Type: ir.PartImage, Media: m}
	}
	return ir.Part{Type: ir.PartFile, Media: m}
}

func remove(q []string, id string) []string {
	for i, x := range q {
		if x == id {
			return append(q[:i:i], q[i+1:]...)
		}
	}
	return q
}

func sanitize(s string) string {
	var b strings.Builder
	for _, c := range s {
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-' {
			b.WriteRune(c)
		}
	}
	return b.String()
}

// responseText renders a functionResponse.response object as tool output.
func responseText(raw json.RawMessage) string {
	var m map[string]json.RawMessage
	if json.Unmarshal(raw, &m) == nil && len(m) == 1 {
		for _, k := range []string{"result", "output", "content"} {
			if v, ok := m[k]; ok {
				if s, ok := ir.TextOrParts(v); ok {
					return s
				}
				return string(v)
			}
		}
	}
	return string(raw)
}

// lowerTypes converts Gemini's uppercase OpenAPI types ("OBJECT") into JSON
// Schema lowercase types.
func lowerTypes(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return raw
	}
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return raw
	}
	var walk func(any)
	walk = func(x any) {
		switch t := x.(type) {
		case map[string]any:
			if s, ok := t["type"].(string); ok {
				t["type"] = strings.ToLower(s)
			}
			if p, ok := t["properties"].(map[string]any); ok {
				for _, pv := range p {
					walk(pv)
				}
			}
			if it, ok := t["items"]; ok {
				walk(it)
			}
			if a, ok := t["anyOf"].([]any); ok {
				for _, e := range a {
					walk(e)
				}
			}
		}
	}
	walk(v)
	b, _ := json.Marshal(v)
	return b
}

// ---- encode ----

func isGemini3(model string) bool { return strings.Contains(strings.ToLower(model), "gemini-3") }

// needsSignatures reports whether the model validates thought signatures on
// function calls (Gemini 3+); older models ignore them.
func needsSignatures(model string) bool {
	m := strings.ToLower(model)
	return !strings.Contains(m, "gemini-1") && !strings.Contains(m, "gemini-2")
}

func (Codec) EncodeRequest(r *ir.Request) ([]byte, error) {
	out := obj{}
	var sys []obj
	for _, p := range r.System {
		if p.Text != "" {
			sys = append(sys, obj{"text": p.Text})
		}
	}
	if len(sys) > 0 {
		out["systemInstruction"] = obj{"parts": sys}
	}

	contents := []obj{}
	for _, m := range ir.RepairToolPairing(r.Messages, &r.Diags) {
		parts := encodeParts(m, r)
		if len(parts) == 0 {
			continue
		}
		role := "user"
		if m.Role == ir.RoleAssistant {
			role = "model"
			if needsSignatures(r.Model) {
				ensureSignature(parts)
			}
		}
		contents = append(contents, obj{"role": role, "parts": parts})
	}
	out["contents"] = contents

	if len(r.Tools) > 0 {
		decls := make([]obj, 0, len(r.Tools))
		for _, t := range r.Tools {
			d := obj{"name": t.Name}
			if t.Description != "" {
				d["description"] = t.Description
			}
			schema, err := CleanSchema(ir.SchemaOrEmpty(t.Parameters))
			if err != nil {
				r.Warn("invalid_tool_schema", "tool "+t.Name+": "+err.Error())
			} else if !schemaIsEmpty(schema) {
				d["parameters"] = schema
			}
			decls = append(decls, d)
		}
		out["tools"] = []obj{{"functionDeclarations": decls}}
		if tc := r.ToolChoice; tc != nil {
			fc := obj{}
			switch tc.Mode {
			case ir.ToolChoiceAuto:
				fc["mode"] = "AUTO"
			case ir.ToolChoiceNone:
				fc["mode"] = "NONE"
			case ir.ToolChoiceRequired:
				fc["mode"] = "ANY"
			case ir.ToolChoiceTool:
				fc["mode"] = "ANY"
				fc["allowedFunctionNames"] = []string{tc.Name}
			}
			out["toolConfig"] = obj{"functionCallingConfig": fc}
		}
	}

	gc := obj{}
	if r.Temperature != nil {
		gc["temperature"] = *r.Temperature
	}
	if r.TopP != nil {
		gc["topP"] = *r.TopP
	}
	if r.TopK != nil {
		gc["topK"] = *r.TopK
	}
	if r.MaxTokens != nil {
		gc["maxOutputTokens"] = *r.MaxTokens
	}
	if len(r.Stop) > 0 {
		stops := r.Stop
		if len(stops) > 5 {
			r.Info("stop_truncated", "gemini supports at most 5 stop sequences")
			stops = stops[:5]
		}
		gc["stopSequences"] = stops
	}
	if r.Seed != nil {
		gc["seed"] = *r.Seed
	}
	if rf := r.ResponseFormat; rf != nil && (rf.Type == "json_object" || rf.Type == "json_schema") {
		gc["responseMimeType"] = "application/json"
		if len(rf.Schema) > 0 {
			if s, err := CleanSchema(rf.Schema); err == nil {
				gc["responseSchema"] = s
			}
		}
	}
	if rs := r.Reasoning; rs != nil {
		tc := obj{}
		effort := rs.EffortLevel()
		switch {
		case effort == "none":
			if !strings.Contains(strings.ToLower(r.Model), "pro") {
				tc["thinkingBudget"] = 0
			}
		case isGemini3(r.Model):
			level := "high"
			if effort == "minimal" || effort == "low" {
				level = "low"
			}
			tc["thinkingLevel"] = level
			tc["includeThoughts"] = true
		default:
			tc["thinkingBudget"] = rs.Budget()
			tc["includeThoughts"] = true
		}
		if len(tc) > 0 {
			gc["thinkingConfig"] = tc
		}
	}
	if len(gc) > 0 {
		out["generationConfig"] = gc
	}
	return json.Marshal(out)
}

// ensureSignature gives the first function call of a model turn a
// placeholder signature when none is known.
func ensureSignature(parts []obj) {
	for _, p := range parts {
		if _, ok := p["functionCall"]; ok {
			if _, has := p["thoughtSignature"]; !has {
				p["thoughtSignature"] = SkipSignature
			}
			return
		}
	}
}

func encodeParts(m ir.Message, r *ir.Request) []obj {
	var out []obj
	for _, p := range m.Parts {
		switch p.Type {
		case ir.PartText:
			if p.Text != "" {
				out = append(out, obj{"text": p.Text})
			}
		case ir.PartImage, ir.PartFile:
			if o := mediaObj(p.Media, r); o != nil {
				out = append(out, o)
			}
		case ir.PartThinking:
			if m.Role == ir.RoleAssistant && p.SigOrigin == self && p.Signature != "" {
				out = append(out, obj{"text": p.Text, "thought": true, "thoughtSignature": p.Signature})
			}
		case ir.PartToolCall:
			fc := obj{"functionCall": obj{"name": p.ToolName, "args": ir.ArgsObject(p.Arguments, &r.Diags)}}
			if p.SigOrigin == self && p.Signature != "" {
				fc["thoughtSignature"] = p.Signature
			}
			out = append(out, fc)
		case ir.PartToolResult:
			out = append(out, obj{"functionResponse": obj{"name": p.ToolName, "response": resultObj(p)}})
			for _, c := range p.Content {
				if c.Type == ir.PartImage || c.Type == ir.PartFile {
					if o := mediaObj(c.Media, r); o != nil {
						out = append(out, o)
					}
				}
			}
		}
	}
	return out
}

func resultObj(p ir.Part) json.RawMessage {
	text := ir.TextOf(p.Content)
	key := "result"
	if p.IsError {
		key = "error"
	}
	if t := strings.TrimSpace(text); strings.HasPrefix(t, "{") && json.Valid([]byte(t)) && !p.IsError {
		return json.RawMessage(t)
	}
	b, _ := json.Marshal(obj{key: text})
	return b
}

func mediaObj(m *ir.Media, r *ir.Request) obj {
	if m == nil {
		return nil
	}
	if m.Data != "" {
		mt := m.MediaType
		if mt == "" {
			mt = "application/octet-stream"
		}
		return obj{"inlineData": obj{"mimeType": mt, "data": m.Data}}
	}
	if mt, data, ok := ir.ParseDataURL(m.URL); ok {
		return obj{"inlineData": obj{"mimeType": mt, "data": data}}
	}
	if m.URL != "" {
		mt := m.MediaType
		if mt == "" {
			mt = guessMime(m.URL)
		}
		return obj{"fileData": obj{"mimeType": mt, "fileUri": m.URL}}
	}
	r.Warn("unsupported_media_source", "media part without data or url dropped")
	return nil
}

func guessMime(u string) string {
	l := strings.ToLower(u)
	switch {
	case strings.HasSuffix(l, ".png"):
		return "image/png"
	case strings.HasSuffix(l, ".gif"):
		return "image/gif"
	case strings.HasSuffix(l, ".webp"):
		return "image/webp"
	case strings.HasSuffix(l, ".pdf"):
		return "application/pdf"
	}
	return "image/jpeg"
}
