package convert_test

import (
	"strings"
	"testing"

	"llm-one-api/internal/convert"
	_ "llm-one-api/internal/convert/gemini"
	"llm-one-api/internal/convert/ir"
	_ "llm-one-api/internal/convert/openairesponses"
	"llm-one-api/internal/convert/sse"
)

const (
	responses = ir.FormatOpenAIResponses
	gemini    = ir.FormatGemini
)

func typedEvents(lines ...string) []sse.Event { return claudeEvents(lines...) }

func TestResponsesToClaudeRequest(t *testing.T) {
	in := `{"model":"gpt-5","instructions":"sys","max_output_tokens":500,"reasoning":{"effort":"low"},
	  "tools":[{"type":"function","name":"f","parameters":{"type":"object","properties":{"a":{"type":"string"}}}},{"type":"web_search"}],
	  "input":[
	    {"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":"data:image/png;base64,QQ=="}]},
	    {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"why"}],"encrypted_content":"llm1:claude:CSIG"},
	    {"type":"function_call","call_id":"call_1","name":"f","arguments":"{\"a\":\"x\"}"},
	    {"type":"function_call_output","call_id":"call_1","output":"done"}
	  ]}`
	out, req, err := convert.Request(responses, claude, []byte(in), ir.LossAllow)
	if err != nil {
		t.Fatal(err)
	}
	m := mustJSON(t, out)
	if m["system"].([]any)[0].(map[string]any)["text"] != "sys" {
		t.Errorf("system: %v", m["system"])
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages: %s", out)
	}
	asst := msgs[1].(map[string]any)["content"].([]any)
	th := asst[0].(map[string]any)
	if th["type"] != "thinking" || th["signature"] != "CSIG" || th["thinking"] != "why" {
		t.Errorf("claude-origin reasoning must be replayed as thinking: %v", th)
	}
	if asst[1].(map[string]any)["type"] != "tool_use" {
		t.Errorf("tool_use: %v", asst[1])
	}
	res := msgs[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if res["type"] != "tool_result" || res["tool_use_id"] != "call_1" || res["content"] != "done" {
		t.Errorf("tool_result: %v", res)
	}
	// signed thinking present -> thinking stays enabled
	if _, ok := m["thinking"]; !ok {
		t.Errorf("thinking should be enabled: %s", out)
	}
	if len(m["tools"].([]any)) != 1 || len(req.Diags) == 0 {
		t.Errorf("hosted tool should be dropped with a diagnostic")
	}
}

func TestChatToResponsesRequest(t *testing.T) {
	in := `{"model":"gpt-5","temperature":0.3,"max_tokens":5,"reasoning_effort":"high","stream":true,
	  "messages":[{"role":"system","content":"s"},{"role":"user","content":"q"},
	    {"role":"assistant","content":"let me","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
	    {"role":"tool","tool_call_id":"c1","content":"r"}],
	  "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	out, _, err := convert.Request(chat, responses, []byte(in), ir.LossAllow)
	if err != nil {
		t.Fatal(err)
	}
	m := mustJSON(t, out)
	if m["instructions"] != "s" || m["store"] != false || m["max_output_tokens"].(float64) != 16 {
		t.Errorf("envelope: %s", out)
	}
	if _, ok := m["temperature"]; ok {
		t.Errorf("temperature must be dropped for gpt-5")
	}
	items := m["input"].([]any)
	var types []string
	for _, it := range items {
		types = append(types, it.(map[string]any)["type"].(string))
	}
	if got := strings.Join(types, ","); got != "message,message,function_call,function_call_output" {
		t.Errorf("items: %s", got)
	}
	tool := m["tools"].([]any)[0].(map[string]any)
	if tool["name"] != "f" || tool["strict"] != false {
		t.Errorf("tool: %v", tool)
	}
	if m["reasoning"].(map[string]any)["effort"] != "high" {
		t.Errorf("reasoning: %v", m["reasoning"])
	}
}

func TestResponsesStreamToClaude(t *testing.T) {
	up := typedEvents(
		`{"type":"response.created","response":{"id":"resp_1","model":"gpt-5","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"think"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"think"}],"encrypted_content":"ENC"}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hi"}`,
		`{"type":"response.content_part.done","item_id":"msg_1","output_index":1,"content_index":0}`,
		`{"type":"response.output_item.done","output_index":1,"item":{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"Hi"}]}}`,
		`{"type":"response.output_item.added","output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_9","name":"f","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"{\"a\":"}`,
		`{"type":"response.output_item.done","output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_9","name":"f","arguments":"{\"a\":1}"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":40},"output_tokens":20}}}`,
	)
	out, err := convert.Pipe(responses, claude, up, &ir.Request{Model: "claude-x"})
	if err != nil {
		t.Fatal(err)
	}
	var args, text, sig string
	var final map[string]any
	for _, e := range out {
		m := mustJSON(t, []byte(e.Data))
		switch e.Event {
		case "content_block_delta":
			d := m["delta"].(map[string]any)
			switch d["type"] {
			case "input_json_delta":
				args += d["partial_json"].(string)
			case "text_delta":
				text += d["text"].(string)
			case "signature_delta":
				sig = d["signature"].(string)
			}
		case "message_delta":
			final = m
		}
	}
	if args != `{"a":1}` {
		t.Errorf("args must include the .done remainder: %q", args)
	}
	if text != "Hi" {
		t.Errorf("text: %q", text)
	}
	if sig != "llm1:openai_responses:rs_1|ENC" {
		t.Errorf("signature: %q", sig)
	}
	if final["delta"].(map[string]any)["stop_reason"] != "tool_use" {
		t.Errorf("stop: %v", final)
	}
	u := final["usage"].(map[string]any)
	if u["input_tokens"].(float64) != 60 || u["cache_read_input_tokens"].(float64) != 40 {
		t.Errorf("usage: %v", u)
	}
}

func TestClaudeStreamToResponses(t *testing.T) {
	up := claudeEvents(
		`{"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":7}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":2}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`,
		`{"type":"message_stop"}`,
	)
	out, err := convert.Pipe(claude, responses, up, &ir.Request{Model: "gpt-alias"})
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range out {
		names = append(names, e.Event)
	}
	want := "response.created response.in_progress response.output_item.added response.content_part.added response.output_text.delta response.output_text.done response.content_part.done response.output_item.done response.output_item.added response.function_call_arguments.delta response.function_call_arguments.done response.output_item.done response.completed"
	if got := strings.Join(names, " "); got != want {
		t.Fatalf("events\n got: %s\nwant: %s", got, want)
	}
	done := mustJSON(t, []byte(out[len(out)-1].Data))["response"].(map[string]any)
	output := done["output"].([]any)
	if len(output) != 2 || output[1].(map[string]any)["arguments"] != `{"x":2}` || output[1].(map[string]any)["call_id"] != "toolu_1" {
		t.Errorf("final output: %v", output)
	}
	if done["usage"].(map[string]any)["input_tokens"].(float64) != 7 || done["model"] != "gpt-alias" {
		t.Errorf("final: %v", done)
	}
	seq := mustJSON(t, []byte(out[1].Data))["sequence_number"].(float64)
	if seq != 1 {
		t.Errorf("sequence numbers must increase: %v", seq)
	}
}

func TestResponsesResponseRoundTrip(t *testing.T) {
	in := `{"id":"resp_1","object":"response","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"model":"gpt-5",
	  "output":[{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"partial"}]}],
	  "usage":{"input_tokens":10,"output_tokens":5,"output_tokens_details":{"reasoning_tokens":2}}}`
	out, err := convert.Response(responses, chat, []byte(in), nil)
	if err != nil {
		t.Fatal(err)
	}
	c := mustJSON(t, out)["choices"].([]any)[0].(map[string]any)
	if c["finish_reason"] != "length" || c["message"].(map[string]any)["content"] != "partial" {
		t.Errorf("choice: %v", c)
	}
}

func TestClaudeToGeminiRequest(t *testing.T) {
	in := `{"model":"gemini-2.5-pro","max_tokens":2000,"thinking":{"type":"enabled","budget_tokens":4096},
	  "system":"sys",
	  "tools":[{"name":"lookup","input_schema":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","additionalProperties":false,
	     "properties":{"q":{"type":["string","null"],"format":"uri"},"n":{"type":"integer","const":3},"opts":{"$ref":"#/$defs/O"}},
	     "required":["q","missing"],"$defs":{"O":{"type":"object","properties":{"k":{"type":"string","enum":["a","b"]}}}}}},
	    {"name":"noargs","input_schema":{"type":"object","properties":{}}}],
	  "tool_choice":{"type":"tool","name":"lookup"},
	  "messages":[{"role":"user","content":"q"},
	    {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"lookup","input":{"q":"x"}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"{\"hits\":1}"}]}]}`
	out, _, err := convert.Request(claude, gemini, []byte(in), ir.LossAllow)
	if err != nil {
		t.Fatal(err)
	}
	m := mustJSON(t, out)
	if m["systemInstruction"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"] != "sys" {
		t.Errorf("system: %v", m["systemInstruction"])
	}
	decls := m["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)
	params := decls[0].(map[string]any)["parameters"].(map[string]any)
	s := string(mustMarshal(t, params))
	for _, bad := range []string{"$schema", "additionalProperties", "$ref", "$defs", "missing", `"uri"`, "const"} {
		if strings.Contains(s, bad) {
			t.Errorf("schema still contains %s: %s", bad, s)
		}
	}
	q := params["properties"].(map[string]any)["q"].(map[string]any)
	if q["type"] != "string" || q["nullable"] != true {
		t.Errorf("type array -> nullable: %v", q)
	}
	if params["properties"].(map[string]any)["opts"].(map[string]any)["properties"] == nil {
		t.Errorf("$ref not resolved: %s", s)
	}
	if _, ok := decls[1].(map[string]any)["parameters"]; ok {
		t.Errorf("empty schema must be omitted")
	}
	fcc := m["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" || fcc["allowedFunctionNames"].([]any)[0] != "lookup" {
		t.Errorf("toolConfig: %v", fcc)
	}
	contents := m["contents"].([]any)
	model := contents[1].(map[string]any)
	if model["role"] != "model" {
		t.Errorf("role: %v", model)
	}
	fr := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if fr["name"] != "lookup" || fr["response"].(map[string]any)["hits"].(float64) != 1 {
		t.Errorf("functionResponse: %v", fr)
	}
	tc := m["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
	if tc["thinkingBudget"].(float64) != 4096 || tc["includeThoughts"] != true {
		t.Errorf("thinkingConfig: %v", tc)
	}
}

func TestGemini3PlaceholderSignature(t *testing.T) {
	in := `{"model":"gemini-3-pro-preview","messages":[{"role":"user","content":"q"},
	  {"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"c2","type":"function","function":{"name":"f","arguments":"{}"}}]},
	  {"role":"tool","tool_call_id":"c1","content":"a"},{"role":"tool","tool_call_id":"c2","content":"b"}]}`
	out, _, err := convert.Request(chat, gemini, []byte(in), ir.LossAllow)
	if err != nil {
		t.Fatal(err)
	}
	parts := mustJSON(t, out)["contents"].([]any)[1].(map[string]any)["parts"].([]any)
	if parts[0].(map[string]any)["thoughtSignature"] != "skip_thought_signature_validator" {
		t.Errorf("first call needs placeholder signature: %v", parts[0])
	}
	if _, ok := parts[1].(map[string]any)["thoughtSignature"]; ok {
		t.Errorf("only the first call gets a signature: %v", parts[1])
	}
}

func TestGeminiStreamToChatAndSignatureRoundTrip(t *testing.T) {
	up := dataEvents(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"pondering","thought":true}]}}],"responseId":"r1","modelVersion":"gemini-3-pro"}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi "}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"x":1}},"thoughtSignature":"R1NJRw=="}]},"finishReason":"STOP"}],
		  "usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":10,"thoughtsTokenCount":5,"cachedContentTokenCount":20}}`,
	)
	out, err := convert.Pipe(gemini, chat, up, &ir.Request{Model: "gpt-alias", IncludeUsage: true})
	if err != nil {
		t.Fatal(err)
	}
	var callID, args, finish, reasoning string
	var usage map[string]any
	for _, e := range out {
		if e.Data == "[DONE]" {
			continue
		}
		m := mustJSON(t, []byte(e.Data))
		if u, ok := m["usage"].(map[string]any); ok {
			usage = u
		}
		ch := m["choices"].([]any)
		if len(ch) == 0 {
			continue
		}
		c := ch[0].(map[string]any)
		if f, ok := c["finish_reason"].(string); ok {
			finish = f
		}
		d := c["delta"].(map[string]any)
		if r, ok := d["reasoning_content"].(string); ok {
			reasoning += r
		}
		if tcs, ok := d["tool_calls"].([]any); ok {
			tc := tcs[0].(map[string]any)
			if id, ok := tc["id"].(string); ok {
				callID = id
			}
			args += tc["function"].(map[string]any)["arguments"].(string)
		}
	}
	if finish != "tool_calls" || args != `{"x":1}` || reasoning != "pondering" {
		t.Errorf("finish=%q args=%q reasoning=%q", finish, args, reasoning)
	}
	if usage["prompt_tokens"].(float64) != 50 || usage["completion_tokens"].(float64) != 15 ||
		usage["prompt_tokens_details"].(map[string]any)["cached_tokens"].(float64) != 20 {
		t.Errorf("usage: %v", usage)
	}
	// The client echoes the packed id back; the Gemini request must carry the signature.
	next := `{"model":"gemini-3-pro","messages":[{"role":"user","content":"q"},
	  {"role":"assistant","tool_calls":[{"id":"` + callID + `","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]},
	  {"role":"tool","tool_call_id":"` + callID + `","content":"ok"}]}`
	body, _, err := convert.Request(chat, gemini, []byte(next), ir.LossAllow)
	if err != nil {
		t.Fatal(err)
	}
	part := mustJSON(t, body)["contents"].([]any)[1].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if part["thoughtSignature"] != "R1NJRw==" {
		t.Errorf("gemini signature lost in round trip: %v", part)
	}
}

func TestGeminiClientRequestSnakeCaseAndPairing(t *testing.T) {
	in := `{"system_instruction":{"parts":[{"text":"sys"}]},
	  "contents":[{"role":"user","parts":[{"text":"q"}]},
	    {"role":"model","parts":[{"function_call":{"name":"f","args":{"user_id":1}}},{"functionCall":{"name":"f","args":{}}}]},
	    {"role":"user","parts":[{"functionResponse":{"name":"f","response":{"result":"one"}}},{"functionResponse":{"name":"f","response":{"result":"two"}}}]}],
	  "generation_config":{"max_output_tokens":99,"thinking_config":{"thinking_budget":0}}}`
	req, err := convert.MustGet(gemini).DecodeRequest([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	if ir.TextOf(req.System) != "sys" || req.MaxTokens == nil || *req.MaxTokens != 99 || req.Reasoning.Effort != "none" {
		t.Errorf("snake_case not normalized: %+v", req)
	}
	calls := req.Messages[1].Parts
	results := req.Messages[2].Parts
	if calls[0].Arguments != `{"user_id":1}` {
		t.Errorf("args keys must not be rewritten: %s", calls[0].Arguments)
	}
	if results[0].ToolCallID != calls[0].ToolCallID || results[1].ToolCallID != calls[1].ToolCallID || calls[0].ToolCallID == calls[1].ToolCallID {
		t.Errorf("pairing by order failed: %+v / %+v", calls, results)
	}
	if ir.TextOf(results[1].Content) != "two" {
		t.Errorf("result text: %+v", results[1])
	}
}

func TestClaudeStreamToGemini(t *testing.T) {
	up := claudeEvents(
		`{"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"yo"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"true}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}`,
		`{"type":"message_stop"}`,
	)
	out, err := convert.Pipe(claude, gemini, up, &ir.Request{Model: "gemini-alias"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("want text chunk, call chunk, final chunk; got %d:\n%s", len(out), joinSSE(out))
	}
	call := mustJSON(t, []byte(out[1].Data))["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)
	fc := call["functionCall"].(map[string]any)
	if fc["name"] != "f" || fc["args"].(map[string]any)["a"] != true {
		t.Errorf("functionCall: %v", fc)
	}
	last := mustJSON(t, []byte(out[2].Data))
	if last["candidates"].([]any)[0].(map[string]any)["finishReason"] != "STOP" || last["usageMetadata"].(map[string]any)["promptTokenCount"].(float64) != 3 {
		t.Errorf("final chunk: %v", last)
	}
}

func TestEveryPairConvertsBasicRequest(t *testing.T) {
	bodies := map[ir.Format]string{
		chat:      `{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		claude:    `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`,
		responses: `{"model":"m","input":"hi"}`,
		gemini:    `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`,
	}
	for from, body := range bodies {
		for to := range bodies {
			out, _, err := convert.Request(from, to, []byte(body), ir.LossAllow)
			if err != nil {
				t.Errorf("%s -> %s: %v", from, to, err)
				continue
			}
			if !strings.Contains(string(out), "hi") {
				t.Errorf("%s -> %s lost the prompt: %s", from, to, out)
			}
		}
	}
}
