package relay

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"llm-one-api/internal/convert"
	"llm-one-api/internal/convert/ir"
	"llm-one-api/internal/convert/sse"
)

func TestMatchModel(t *testing.T) {
	cases := []struct {
		pattern, model string
		want           bool
	}{
		{"*", "anything", true},
		{"claude-*", "claude-3", true},
		{"claude-*", "gpt-4", false},
		{"*-mini", "gpt-4o-mini", true},
		{"gpt-*-turbo", "gpt-4-turbo", true},
		{"gpt-*-turbo", "gpt-4", false},
		{"exact", "exact", true},
	}
	for _, tc := range cases {
		if got := matchModel(tc.pattern, tc.model); got != tc.want {
			t.Errorf("matchModel(%q, %q) = %v, want %v", tc.pattern, tc.model, got, tc.want)
		}
	}
}

func TestParseConfigExpandsEnv(t *testing.T) {
	t.Setenv("TEST_UPSTREAM_KEY", "secret-key")
	cfg, err := ParseConfig(strings.NewReader(`
listen: ":9"
loss: strict
timeout: 1s
upstreams:
  - name: only
    format: claude
    base_url: https://example.com/v1
    api_key: ${TEST_UPSTREAM_KEY}
routes:
  - model: claude-*
    upstream: only
    upstream_model: claude-real
`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Upstreams[0].APIKey != "secret-key" || cfg.Loss != ir.LossStrict || cfg.Timeout != time.Second {
		t.Fatalf("config: %+v", cfg)
	}
	rt, up, ok := cfg.match("claude-x")
	if !ok || up.Name != "only" || rt.UpstreamModel != "claude-real" {
		t.Fatalf("match: %+v %+v %v", rt, up, ok)
	}
	if _, _, ok := cfg.match("gpt"); ok {
		t.Fatal("gpt should not match")
	}
}

func TestUpstreamURL(t *testing.T) {
	u, err := upstreamURL(Upstream{Format: ir.FormatOpenAIChat, BaseURL: "https://api.openai.com/v1"}, "gpt", false)
	if err != nil || u != "https://api.openai.com/v1/chat/completions" {
		t.Fatalf("openai: %s %v", u, err)
	}
	u, err = upstreamURL(Upstream{Format: ir.FormatGemini, BaseURL: "https://generativelanguage.googleapis.com"}, "gemini-2.0-flash", true)
	if err != nil || u != "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse" {
		t.Fatalf("gemini: %s %v", u, err)
	}
}

func TestStreamMatchesPipe(t *testing.T) {
	up := []sse.Event{
		{Data: `{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{"content":"Hel"}}]}`},
		{Data: `{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{"content":"lo"}}]}`},
		{Data: `{"id":"c1","model":"gpt","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`},
		{Data: `{"id":"c1","model":"gpt","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}`},
		{Data: "[DONE]"},
	}
	var raw bytes.Buffer
	for _, ev := range up {
		raw.Write(ev.Bytes())
	}
	client := &ir.Request{Model: "claude-alias"}
	var got bytes.Buffer
	if err := Stream(ir.FormatOpenAIChat, ir.FormatClaude, bytes.NewReader(raw.Bytes()), &got, nil, client); err != nil {
		t.Fatal(err)
	}
	want, err := convert.Pipe(ir.FormatOpenAIChat, ir.FormatClaude, up, client)
	if err != nil {
		t.Fatal(err)
	}
	var wantBuf bytes.Buffer
	for _, ev := range want {
		wantBuf.Write(ev.Bytes())
	}
	if normalizeStream(got.String()) != normalizeStream(wantBuf.String()) {
		t.Fatalf("stream\n got: %s\nwant: %s", got.String(), wantBuf.String())
	}
}

var msgID = regexp.MustCompile(`"id":"msg_[0-9a-f]+"`)

func normalizeStream(s string) string { return msgID.ReplaceAllString(s, `"id":"msg"`) }

func TestRelayChatToClaude(t *testing.T) {
	var gotPath, gotKey, gotVer string
	var gotBody []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.RequestURI()
		gotKey = r.Header.Get("x-api-key")
		gotVer = r.Header.Get("anthropic-version")
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","model":"claude-real","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}`)
	}))
	defer up.Close()

	cfg := testConfig(up.URL, ir.FormatClaude, "alias-*", "claude-real")
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"alias-a","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer client-key")
	New(cfg).Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	if gotPath != "/v1/messages" || gotKey != "gateway-key" || gotVer != "2023-06-01" {
		t.Fatalf("upstream path=%s key=%s ver=%s", gotPath, gotKey, gotVer)
	}
	var sent map[string]any
	if err := json.Unmarshal(gotBody, &sent); err != nil {
		t.Fatal(err)
	}
	if sent["model"] != "claude-real" {
		t.Fatalf("upstream model: %v", sent["model"])
	}
	out := mustJSON(t, rec.Body.Bytes())
	if out["model"] != "alias-a" {
		t.Fatalf("client model: %v", out["model"])
	}
	msg := out["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "hello" {
		t.Fatalf("content: %v", msg)
	}
}

func TestRelayStreamAndJSONFallback(t *testing.T) {
	const claudeSSE = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.Header.Get("Accept"), "text/event-stream") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, claudeSSE)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","model":"claude","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer up.Close()
	h := New(testConfig(up.URL, ir.FormatClaude, "*", "")).Handler()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	h.ServeHTTP(rec, req)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"content":"hi"`) || !strings.Contains(rec.Body.String(), "[DONE]") {
		t.Fatalf("sse relay: %d %s", rec.Code, rec.Body.String())
	}
}

func TestJSONUpstreamSynthesizedForStreamClient(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","model":"claude","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`)
	}))
	defer up.Close()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	New(testConfig(up.URL, ir.FormatClaude, "*", "")).Handler().ServeHTTP(rec, req)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"content":"hello"`) || !strings.HasSuffix(strings.TrimSpace(rec.Body.String()), "[DONE]") {
		t.Fatalf("synthesized stream: %d %s", rec.Code, rec.Body.String())
	}
}

func TestStrictLossDoesNotCallUpstream(t *testing.T) {
	called := false
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(500)
	}))
	defer up.Close()
	cfg := testConfig(up.URL, ir.FormatOpenAIChat, "*", "")
	cfg.Loss = ir.LossStrict
	rec := httptest.NewRecorder()
	body := `{"model":"m","max_tokens":20,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	New(cfg).Handler().ServeHTTP(rec, req)
	if called {
		t.Fatal("upstream was called")
	}
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "hosted_tool_dropped") {
		t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
	}
}

func TestGeminiPath(t *testing.T) {
	var gotPath string
	var gotModel string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		var m map[string]any
		_ = json.NewDecoder(r.Body).Decode(&m)
		gotModel, _ = m["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","model":"gpt","choices":[{"message":{"role":"assistant","content":"yo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer up.Close()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.0-flash:generateContent", strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
	New(testConfig(up.URL, ir.FormatOpenAIChat, "gemini-*", "")).Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	if gotPath != "/v1/chat/completions" || gotModel != "gemini-2.0-flash" {
		t.Fatalf("path %s model %s", gotPath, gotModel)
	}
	if !strings.Contains(rec.Body.String(), "yo") {
		t.Fatalf("body %s", rec.Body.String())
	}
}

func TestHealthz(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	New(testConfig("http://127.0.0.1", ir.FormatOpenAIChat, "*", "")).Handler().ServeHTTP(rec, req)
	if rec.Code != 200 || rec.Body.String() != "ok\n" {
		t.Fatalf("health: %d %q", rec.Code, rec.Body.String())
	}
}

func testConfig(base string, format ir.Format, pattern, upstreamModel string) Config {
	return Config{
		Listen:  ":0",
		Loss:    ir.LossAllow,
		Timeout: time.Minute,
		Upstreams: []Upstream{{
			Name:    "up",
			Format:  format,
			BaseURL: base,
			APIKey:  "gateway-key",
		}},
		Routes: []Route{{Model: pattern, Upstream: "up", UpstreamModel: upstreamModel}},
	}
}

func mustJSON(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("json: %v %s", err, b)
	}
	return m
}
