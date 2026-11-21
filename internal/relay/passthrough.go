package relay

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"time"

	"llm-one-api/internal/convert/ir"
)

// peekRequest reads only the routing fields, so passthrough never depends on
// the decoder understanding every field of the body.
func peekRequest(body []byte) (model string, stream bool) {
	var p struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	_ = json.Unmarshal(body, &p)
	return p.Model, p.Stream
}

// withModel replaces the top-level "model" field, keeping everything else.
func withModel(body []byte, model string) ([]byte, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, err
	}
	b, _ := json.Marshal(model)
	m["model"] = b
	return json.Marshal(m)
}

// passthrough forwards a request whose client and upstream formats match.
// Only the model name is rewritten; the response bytes are relayed as-is.
func (s *Server) passthrough(w http.ResponseWriter, r *http.Request, format ir.Format, up Upstream, route Route, body []byte, model string, stream bool) int {
	upModel := model
	if route.UpstreamModel != "" {
		upModel = route.UpstreamModel
		if format != ir.FormatGemini { // Gemini carries the model in the URL
			b, err := withModel(body, upModel)
			if err != nil {
				return writeErr(w, format, http.StatusBadRequest, "invalid_request_error", "invalid JSON body")
			}
			body = b
		}
	}
	endpoint, err := upstreamURL(up, upModel, stream)
	if err != nil {
		return writeErr(w, format, http.StatusInternalServerError, "api_error", err.Error())
	}
	ctx := r.Context()
	if s.cfg.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.cfg.Timeout)
		defer cancel()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return writeErr(w, format, http.StatusInternalServerError, "api_error", err.Error())
	}
	req.Header = upstreamHeaders(r, up, stream)
	resp, err := s.http.Do(req)
	if err != nil {
		slog.Error("upstream", "upstream", up.Name, "err", err)
		return writeErr(w, format, http.StatusBadGateway, "api_error", "upstream request failed")
	}
	defer resp.Body.Close()

	for _, h := range []string{"Content-Type", "Retry-After", "request-id", "x-request-id"} {
		if v := resp.Header.Get(h); v != "" {
			w.Header().Set(h, v)
		}
	}
	for k, v := range resp.Header {
		lk := strings.ToLower(k)
		if strings.HasPrefix(lk, "anthropic-ratelimit-") || strings.HasPrefix(lk, "x-ratelimit-") {
			w.Header()[k] = v
		}
	}
	if isEventStream(resp.Header) {
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
	}
	w.WriteHeader(resp.StatusCode)
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				break
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if rerr != nil {
			if rerr != io.EOF {
				slog.Error("passthrough", "upstream", up.Name, "err", rerr)
			}
			break
		}
	}
	return resp.StatusCode
}

// handleModels lists the concrete (non-glob) model names from the routes, in
// OpenAI format, or Anthropic format when the client sends anthropic-version.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if !s.cfg.authorized(clientKey(r)) {
		writeErr(w, ir.FormatOpenAIChat, http.StatusUnauthorized, "authentication_error", "invalid or missing API key")
		return
	}
	seen := map[string]bool{}
	var ids []string
	for _, rt := range s.cfg.Routes {
		if !strings.Contains(rt.Model, "*") && !seen[rt.Model] {
			seen[rt.Model] = true
			ids = append(ids, rt.Model)
		}
	}
	sort.Strings(ids)
	now := time.Now()
	var body any
	if r.Header.Get("anthropic-version") != "" {
		data := make([]map[string]any, 0, len(ids))
		for _, id := range ids {
			data = append(data, map[string]any{"type": "model", "id": id, "display_name": id, "created_at": now.UTC().Format(time.RFC3339)})
		}
		first, last := "", ""
		if len(ids) > 0 {
			first, last = ids[0], ids[len(ids)-1]
		}
		body = map[string]any{"data": data, "has_more": false, "first_id": first, "last_id": last}
	} else {
		data := make([]map[string]any, 0, len(ids))
		for _, id := range ids {
			data = append(data, map[string]any{"id": id, "object": "model", "created": now.Unix(), "owned_by": "llm-one-api"})
		}
		body = map[string]any{"object": "list", "data": data}
	}
	b, _ := json.Marshal(body)
	writeJSON(w, http.StatusOK, b)
}
