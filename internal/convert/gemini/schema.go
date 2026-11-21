package gemini

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---- snake_case -> camelCase normalization of request keys ----
//
// The Gemini REST API accepts both "system_instruction" and
// "systemInstruction". We normalize structural keys to camelCase but never
// descend into user-defined payloads (function args, schemas, responses).

var opaqueKeys = map[string]bool{
	"args": true, "response": true, "parameters": true, "parametersJsonSchema": true,
	"responseSchema": true, "responseJsonSchema": true, "properties": true, "metadata": true,
	"labels": true,
}

func camel(s string) string {
	if !strings.Contains(s, "_") {
		return s
	}
	parts := strings.Split(s, "_")
	for i := 1; i < len(parts); i++ {
		if parts[i] != "" {
			parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
		}
	}
	return strings.Join(parts, "")
}

func normalizeKeys(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			ck := camel(k)
			if opaqueKeys[ck] {
				out[ck] = val
			} else {
				out[ck] = normalizeKeys(val)
			}
		}
		return out
	case []any:
		for i := range t {
			t[i] = normalizeKeys(t[i])
		}
		return t
	}
	return v
}

// normalizeBody rewrites snake_case keys to camelCase.
func normalizeBody(body []byte) ([]byte, error) {
	var v any
	dec := json.NewDecoder(strings.NewReader(string(body)))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	return json.Marshal(normalizeKeys(v))
}

// ---- JSON Schema -> Gemini (OpenAPI 3 subset) schema ----

var allowedSchemaKeys = map[string]bool{
	"type": true, "format": true, "title": true, "description": true, "nullable": true, "enum": true,
	"maxItems": true, "minItems": true, "properties": true, "required": true, "minProperties": true,
	"maxProperties": true, "minLength": true, "maxLength": true, "pattern": true, "example": true,
	"anyOf": true, "propertyOrdering": true, "default": true, "items": true, "minimum": true, "maximum": true,
}

// CleanSchema converts a JSON Schema into the subset accepted by Gemini's
// "parameters" field: resolves local $refs, folds type arrays into nullable,
// turns const into enum, and drops unsupported keywords.
func CleanSchema(raw json.RawMessage) (map[string]any, error) {
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, fmt.Errorf("invalid schema: %w", err)
	}
	defs := map[string]any{}
	for _, k := range []string{"$defs", "definitions"} {
		if d, ok := root[k].(map[string]any); ok {
			for name, v := range d {
				defs["#/"+k+"/"+name] = v
			}
		}
	}
	out, _ := clean(root, defs, 0).(map[string]any)
	return out, nil
}

func clean(v any, defs map[string]any, depth int) any {
	m, ok := v.(map[string]any)
	if !ok || depth > 32 {
		return v
	}
	if ref, ok := m["$ref"].(string); ok {
		if target, ok := defs[ref]; ok && depth < 16 {
			merged := map[string]any{}
			if tm, ok := target.(map[string]any); ok {
				for k, val := range tm {
					merged[k] = val
				}
			}
			if d, ok := m["description"]; ok {
				merged["description"] = d
			}
			return clean(merged, defs, depth+1)
		}
		return map[string]any{"type": "object"}
	}
	out := map[string]any{}
	// type: ["string","null"] -> type: string, nullable: true
	switch t := m["type"].(type) {
	case string:
		out["type"] = t
	case []any:
		for _, x := range t {
			if s, _ := x.(string); s == "null" {
				out["nullable"] = true
			} else if s != "" && out["type"] == nil {
				out["type"] = s
			}
		}
	}
	if c, ok := m["const"]; ok {
		out["enum"] = []any{fmt.Sprint(c)}
	}
	// allOf with a single entry is common in generated schemas; flatten it.
	if all, ok := m["allOf"].([]any); ok && len(all) == 1 {
		if am, ok := clean(all[0], defs, depth+1).(map[string]any); ok {
			for k, val := range am {
				if _, exists := out[k]; !exists {
					out[k] = val
				}
			}
		}
	}
	for k, val := range m {
		if !allowedSchemaKeys[k] || k == "type" {
			continue
		}
		switch k {
		case "properties":
			props, ok := val.(map[string]any)
			if !ok {
				continue
			}
			cp := map[string]any{}
			for name, pv := range props {
				cp[name] = clean(pv, defs, depth+1)
			}
			out["properties"] = cp
		case "items":
			out["items"] = clean(val, defs, depth+1)
		case "anyOf":
			arr, ok := val.([]any)
			if !ok {
				continue
			}
			var opts []any
			for _, o := range arr {
				if om, ok := o.(map[string]any); ok && om["type"] == "null" {
					out["nullable"] = true
					continue
				}
				opts = append(opts, clean(o, defs, depth+1))
			}
			if len(opts) == 1 {
				if om, ok := opts[0].(map[string]any); ok {
					for k2, v2 := range om {
						if _, exists := out[k2]; !exists {
							out[k2] = v2
						}
					}
				}
			} else if len(opts) > 1 {
				out["anyOf"] = opts
			}
		case "enum":
			arr, ok := val.([]any)
			if !ok {
				continue
			}
			strs := make([]any, 0, len(arr))
			for _, e := range arr {
				if e == nil {
					out["nullable"] = true
					continue
				}
				strs = append(strs, fmt.Sprint(e))
			}
			out["enum"] = strs
			if out["type"] == nil || out["type"] != "string" {
				out["type"] = "string"
			}
		case "format":
			f, _ := val.(string)
			switch out["type"] {
			case "string":
				if f == "enum" || f == "date-time" {
					out["format"] = f
				}
			case "integer":
				if f == "int32" || f == "int64" {
					out["format"] = f
				}
			case "number":
				if f == "float" || f == "double" {
					out["format"] = f
				}
			}
		default:
			out[k] = val
		}
	}
	// required must reference existing properties
	if req, ok := out["required"].([]any); ok {
		props, _ := out["properties"].(map[string]any)
		var keep []any
		for _, r := range req {
			if s, ok := r.(string); ok && props[s] != nil {
				keep = append(keep, s)
			}
		}
		if len(keep) == 0 {
			delete(out, "required")
		} else {
			out["required"] = keep
		}
	}
	if out["type"] == nil {
		switch {
		case out["properties"] != nil:
			out["type"] = "object"
		case out["items"] != nil:
			out["type"] = "array"
		case out["anyOf"] == nil && out["enum"] == nil:
			out["type"] = "string"
		}
	}
	if out["type"] == "array" && out["items"] == nil {
		out["items"] = map[string]any{"type": "string"}
	}
	return out
}

// schemaIsEmpty reports whether a tool schema declares no properties.
func schemaIsEmpty(s map[string]any) bool {
	if s == nil {
		return true
	}
	props, _ := s["properties"].(map[string]any)
	return len(props) == 0
}
