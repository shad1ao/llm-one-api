package convert_test

import (
	"encoding/json"
	"strings"
	"testing"

	"llm-one-api/internal/convert"
	_ "llm-one-api/internal/convert/claude"
	"llm-one-api/internal/convert/ir"
	_ "llm-one-api/internal/convert/openaichat"
	"llm-one-api/internal/convert/sse"
)

const (
	chat   = ir.FormatOpenAIChat
	claude = ir.FormatClaude
)

func mustJSON(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, b)
	}
	return m
}

func dataEvents(lines ...string) []sse.Event {
	out := make([]sse.Event, len(lines))
	for i, l := range lines {
		out[i] = sse.Event{Data: l}
	}
	return out
}

func claudeEvents(lines ...string) []sse.Event {
	out := make([]sse.Event, len(lines))
	for i, l := range lines {
		var h struct{ Type string }
		_ = json.Unmarshal([]byte(l), &h)
		out[i] = sse.Event{Event: h.Type, Data: l}
	}
	return out
}

func TestClaudeToChatRequest(t *testing.T) {
	in := `{
	  "model":"claude-x","max_tokens":1000,"stream":true,
	  "system":[{"type":"text","text":"be nice","cache_control":{"type":"ephemeral"}}],
	  "tools":[{"name":"get_weather","description":"w","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}},
	           {"type":"web_search_20250305","name":"web_search"}],
	  "tool_choice":{"type":"any"},
	  "messages":[
	    {"role":"user","content":"weather in Paris?"},
	    {"role":"assistant","content":[
	      {"type":"thinking","thinking":"let me check","signature":"sig123"},
	      {"type":"text","text":"Checking."},
	      {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
	    {"role":"user","content":[
	      {"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"sunny"},
	        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]},
	      {"type":"text","text":"thanks"}]}
	  ]}`
	out, req, err := convert.Request(claude, chat, []byte(in), ir.LossAllow)
	if err != nil {
		t.Fatal(err)
	}
	m := mustJSON(t, out)
	msgs := m["messages"].([]any)
	if len(msgs) != 5 {
		t.Fatalf("want 5 messages (system,user,assistant,tool,user), got %d: %s", len(msgs), out)
	}
	if msgs[0].(map[string]any)["role"] != "system" || msgs[0].(map[string]any)["content"] != "be nice" {
		t.Errorf("system: %v", msgs[0])
	}
	asst := msgs[2].(map[string]any)
	tc := asst["tool_calls"].([]any)[0].(map[string]any)
	if tc["id"] != "toolu_1" || tc["function"].(map[string]any)["arguments"] != `{"city":"Paris"}` {
		t.Errorf("tool call: %v", tc)
	}
	if asst["reasoning_content"] != "let me check" {
		t.Errorf("reasoning_content missing: %v", asst)
	}
	tool := msgs[3].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "toolu_1" || tool["content"] != "sunny" {
		t.Errorf("tool msg: %v", tool)
	}
	user := msgs[4].(map[string]any)
	parts := user["content"].([]any)
	if len(parts) != 3 { // lifted-images note, image, "thanks"
		t.Errorf("lifted image user msg: %v", user)
	}
	if m["tool_choice"] != "required" {
		t.Errorf("tool_choice: %v", m["tool_choice"])
	}
	if len(m["tools"].([]any)) != 1 {
		t.Errorf("hosted tool should be dropped")
	}
	if m["stream_options"].(map[string]any)["include_usage"] != true {
		t.Errorf("include_usage not set")
	}
	if len(req.Diags) == 0 {
		t.Errorf("expected diagnostics for dropped hosted tool")
	}
}

func TestChatToClaudeRequest(t *testing.T) {
	in := `{
	  "model":"gpt","temperature":0.5,"stop":"END","reasoning_effort":"high","max_tokens":4000,
	  "messages":[
	    {"role":"system","content":"sys"},
	    {"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,BBBB"}}]},
	    {"role":"assistant","content":null,"tool_calls":[
	       {"id":"call_a","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}},
	       {"id":"call_b","type":"function","function":{"name":"g","arguments":""}}]},
	    {"role":"tool","tool_call_id":"call_a","content":"ra"}
	  ],
	  "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}},
	           {"type":"function","function":{"name":"g"}}]
	}`
	out, req, err := convert.Request(chat, claude, []byte(in), ir.LossAllow)
	if err != nil {
		t.Fatal(err)
	}
	m := mustJSON(t, out)
	if m["system"].([]any)[0].(map[string]any)["text"] != "sys" {
		t.Errorf("system: %v", m["system"])
	}
	// In-progress tool turn without Claude-signed thinking: thinking must be off.
	if _, ok := m["thinking"]; ok {
		t.Errorf("thinking should be disabled for unsigned tool turn")
	}
	if m["temperature"].(float64) != 0.5 {
		t.Errorf("temperature: %v", m["temperature"])
	}
	msgs := m["messages"].([]any)
	img := msgs[0].(map[string]any)["content"].([]any)[1].(map[string]any)
	if img["source"].(map[string]any)["type"] != "base64" || img["source"].(map[string]any)["media_type"] != "image/jpeg" {
		t.Errorf("image: %v", img)
	}
	asst := msgs[1].(map[string]any)["content"].([]any)
	if asst[1].(map[string]any)["input"] == nil {
		t.Errorf("empty args should become {}")
	}
	results := msgs[2].(map[string]any)["content"].([]any)
	if len(results) != 2 {
		t.Fatalf("missing tool result should be synthesized: %v", results)
	}
	if results[1].(map[string]any)["is_error"] != true {
		t.Errorf("placeholder should be error: %v", results[1])
	}
	if m["stop_sequences"].([]any)[0] != "END" {
		t.Errorf("stop: %v", m["stop_sequences"])
	}
	if len(req.Diags) == 0 {
		t.Errorf("expected diagnostics (missing tool result, thinking disabled)")
	}
}

func TestChatToClaudeThinking(t *testing.T) {
	in := `{"model":"gpt","temperature":0.5,"top_k":3,"reasoning_effort":"high","max_tokens":4000,
	  "messages":[{"role":"user","content":"hi"}]}`
	out, _, err := convert.Request(chat, claude, []byte(in), ir.LossAllow)
	if err != nil {
		t.Fatal(err)
	}
	m := mustJSON(t, out)
	th := m["thinking"].(map[string]any)
	if th["type"] != "enabled" || th["budget_tokens"].(float64) != 16384 {
		t.Errorf("thinking: %v", th)
	}
	if m["max_tokens"].(float64) <= 16384 {
		t.Errorf("max_tokens must exceed budget: %v", m["max_tokens"])
	}
	if _, ok := m["temperature"]; ok {
		t.Errorf("temperature must be dropped with thinking")
	}
	if _, ok := m["top_k"]; ok {
		t.Errorf("top_k must be dropped with thinking")
	}
}

func TestThinkingDisabledForUnsignedToolTurn(t *testing.T) {
	in := `{"model":"m","reasoning_effort":"low","messages":[
	  {"role":"user","content":"q"},
	  {"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
	  {"role":"tool","tool_call_id":"c1","content":"r"}],
	  "tools":[{"type":"function","function":{"name":"f"}}]}`
	out, _, err := convert.Request(chat, claude, []byte(in), ir.LossAllow)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := mustJSON(t, out)["thinking"]; ok {
		t.Errorf("thinking should be disabled: %s", out)
	}
}

func TestClaudeResponseToChat(t *testing.T) {
	in := `{"id":"msg_1","type":"message","model":"claude-x","content":[
	  {"type":"thinking","thinking":"hmm","signature":"s"},
	  {"type":"text","text":"hello"},
	  {"type":"tool_use","id":"toolu_9","name":"f","input":{"a":1}}],
	  "stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":100,"cache_creation_input_tokens":20}}`
	out, err := convert.Response(claude, chat, []byte(in), nil)
	if err != nil {
		t.Fatal(err)
	}
	m := mustJSON(t, out)
	c := m["choices"].([]any)[0].(map[string]any)
	if c["finish_reason"] != "tool_calls" {
		t.Errorf("finish: %v", c["finish_reason"])
	}
	msg := c["message"].(map[string]any)
	if msg["content"] != "hello" || msg["reasoning_content"] != "hmm" {
		t.Errorf("message: %v", msg)
	}
	u := m["usage"].(map[string]any)
	if u["prompt_tokens"].(float64) != 130 || u["prompt_tokens_details"].(map[string]any)["cached_tokens"].(float64) != 100 {
		t.Errorf("usage: %v", u)
	}
}

func TestChatResponseToClaudeUsage(t *testing.T) {
	in := `{"id":"x","model":"gpt","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"length"}],
	  "usage":{"prompt_tokens":150,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":100}}}`
	out, err := convert.Response(chat, claude, []byte(in), nil)
	if err != nil {
		t.Fatal(err)
	}
	m := mustJSON(t, out)
	u := m["usage"].(map[string]any)
	if u["input_tokens"].(float64) != 50 || u["cache_read_input_tokens"].(float64) != 100 {
		t.Errorf("claude input_tokens must exclude cache: %v", u)
	}
	if m["stop_reason"] != "max_tokens" {
		t.Errorf("stop: %v", m["stop_reason"])
	}
}

func joinSSE(evs []sse.Event) string {
	var b strings.Builder
	for _, e := range evs {
		b.Write(e.Bytes())
	}
	return b.String()
}

func TestChatStreamToClaude(t *testing.T) {
	up := dataEvents(
		`{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"}}]}`,
		`{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{"content":"Hel"}}]}`,
		`{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		// args fragment before the id is known (deferred start)
		`{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{\"a\""}}]}}]}`,
		`{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":":1}"}}]}}]}`,
		`{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"g","arguments":""}}]}}]}`,
		`{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"id":"c1","model":"gpt","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":9,"prompt_tokens_details":{"cached_tokens":5}}}`,
		`[DONE]`,
	)
	out, err := convert.Pipe(chat, claude, up, &ir.Request{Model: "claude-alias"})
	if err != nil {
		t.Fatal(err)
	}
	var types []string
	var deltaJSON strings.Builder
	var final map[string]any
	for _, e := range out {
		types = append(types, e.Event)
		m := mustJSON(t, []byte(e.Data))
		if e.Event == "content_block_delta" {
			d := m["delta"].(map[string]any)
			if d["type"] == "input_json_delta" && m["index"].(float64) == 2 {
				deltaJSON.WriteString(d["partial_json"].(string))
			}
		}
		if e.Event == "message_delta" {
			final = m
		}
	}
	want := "message_start content_block_start content_block_delta content_block_stop content_block_start content_block_delta content_block_delta content_block_stop content_block_start content_block_delta content_block_delta content_block_stop content_block_start content_block_delta content_block_stop message_delta message_stop"
	if got := strings.Join(types, " "); got != want {
		t.Fatalf("event sequence\n got: %s\nwant: %s\n%s", got, want, joinSSE(out))
	}
	if deltaJSON.String() != `{"a":1}` {
		t.Errorf("tool args: %q", deltaJSON.String())
	}
	if final["delta"].(map[string]any)["stop_reason"] != "tool_use" {
		t.Errorf("stop: %v", final)
	}
	u := final["usage"].(map[string]any)
	if u["input_tokens"].(float64) != 15 || u["cache_read_input_tokens"].(float64) != 5 || u["output_tokens"].(float64) != 9 {
		t.Errorf("usage: %v", u)
	}
	start := mustJSON(t, []byte(out[0].Data))
	if start["message"].(map[string]any)["model"] != "claude-alias" {
		t.Errorf("model should be rewritten to the client's model")
	}
}

func TestClaudeStreamToChat(t *testing.T) {
	up := claudeEvents(
		`{"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":12,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hm"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ok"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"2}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}`,
		`{"type":"message_stop"}`,
	)
	out, err := convert.Pipe(claude, chat, up, &ir.Request{Model: "gpt-alias", IncludeUsage: true})
	if err != nil {
		t.Fatal(err)
	}
	if out[len(out)-1].Data != "[DONE]" {
		t.Fatalf("missing [DONE]")
	}
	var content, reasoning, args strings.Builder
	var finish any
	var usage map[string]any
	for _, e := range out[:len(out)-1] {
		m := mustJSON(t, []byte(e.Data))
		if u, ok := m["usage"].(map[string]any); ok {
			usage = u
		}
		ch := m["choices"].([]any)
		if len(ch) == 0 {
			continue
		}
		c := ch[0].(map[string]any)
		if c["finish_reason"] != nil {
			finish = c["finish_reason"]
		}
		d := c["delta"].(map[string]any)
		if s, ok := d["content"].(string); ok {
			content.WriteString(s)
		}
		if s, ok := d["reasoning_content"].(string); ok {
			reasoning.WriteString(s)
		}
		if tcs, ok := d["tool_calls"].([]any); ok {
			fn := tcs[0].(map[string]any)["function"].(map[string]any)
			args.WriteString(fn["arguments"].(string))
		}
	}
	if content.String() != "ok" || reasoning.String() != "hm" || args.String() != `{"q":2}` || finish != "tool_calls" {
		t.Errorf("content=%q reasoning=%q args=%q finish=%v", content.String(), reasoning.String(), args.String(), finish)
	}
	if usage == nil || usage["prompt_tokens"].(float64) != 12 || usage["completion_tokens"].(float64) != 30 {
		t.Errorf("usage: %v", usage)
	}
}

func TestTruncatedStreamEmitsError(t *testing.T) {
	up := dataEvents(`{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{"content":"partial"}}]}`)
	out, err := convert.Pipe(chat, claude, up, nil)
	if err != nil {
		t.Fatal(err)
	}
	last := out[len(out)-1]
	if last.Event != "error" {
		t.Errorf("truncated stream should end with an error event, got:\n%s", joinSSE(out))
	}
}

func TestThinkingSignatureRoundTrip(t *testing.T) {
	// A foreign signature must be wrapped for the client and only replayed to its origin.
	resp := &ir.Response{Model: "m", Parts: []ir.Part{{Type: ir.PartThinking, Text: "t", Signature: "ENC", SigOrigin: ir.FormatOpenAIResponses}}}
	c := convert.MustGet(claude)
	body, _ := c.EncodeResponse(resp, nil)
	sig := mustJSON(t, body)["content"].([]any)[0].(map[string]any)["signature"].(string)
	if sig != "llm1:openai_responses:ENC" {
		t.Fatalf("wrapped sig: %q", sig)
	}
	req := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"a"},{"role":"assistant","content":[{"type":"thinking","thinking":"t","signature":"` + sig + `"},{"type":"text","text":"b"}]},{"role":"user","content":"c"}]}`
	ireq, err := c.DecodeRequest([]byte(req))
	if err != nil {
		t.Fatal(err)
	}
	p := ireq.Messages[1].Parts[0]
	if p.SigOrigin != ir.FormatOpenAIResponses || p.Signature != "ENC" {
		t.Errorf("unwrap: %+v", p)
	}
	// Re-encoding to Claude must drop the foreign thinking block.
	out, _ := c.EncodeRequest(ireq)
	if strings.Contains(string(out), "thinking") {
		t.Errorf("foreign thinking must not reach Claude: %s", out)
	}
}

func TestCallIDPacking(t *testing.T) {
	id := ir.PackCallID(claude, "call_1", ir.FormatGemini, "gsig/+==")
	base, origin, sig := ir.UnpackCallID(id)
	if base != "call_1" || origin != ir.FormatGemini || sig != "gsig/+==" {
		t.Errorf("unpack: %q %q %q", base, origin, sig)
	}
	if strings.ContainsAny(id, "/+=") {
		t.Errorf("packed id must be url-safe: %q", id)
	}
}

func TestSSEReader(t *testing.T) {
	src := ": keepalive\n\nevent: message_start\ndata: {\"a\":1}\n\ndata: line1\ndata: line2\n\ndata: [DONE]"
	rd := sse.NewReader(strings.NewReader(src))
	var got []sse.Event
	for {
		ev, err := rd.Next()
		if err != nil {
			break
		}
		got = append(got, ev)
	}
	if len(got) != 3 || got[0].Event != "message_start" || got[1].Data != "line1\nline2" || got[2].Data != "[DONE]" {
		t.Errorf("events: %#v", got)
	}
}
