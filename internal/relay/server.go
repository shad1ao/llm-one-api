package relay

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"

	"llm-one-api/internal/convert"
	_ "llm-one-api/internal/convert/all"
	"llm-one-api/internal/convert/ir"
)

const maxBody = 32 << 20

// Server relays a client wire format to a configured upstream.
type Server struct {
	cfg  Config
	http *http.Client
}

// New returns a relay server. The HTTP client has no deadline of its own;
// each request uses Config.Timeout.
func New(cfg Config) *Server {
	return &Server{cfg: cfg, http: &http.Client{}}
}

// Handler is the public HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = io.WriteString(w, "ok\n")
	})
	for _, p := range []string{"/v1/chat/completions", "/chat/completions"} {
		mux.HandleFunc("POST "+p, s.handle(ir.FormatOpenAIChat))
	}
	for _, p := range []string{"/v1/responses", "/responses"} {
		mux.HandleFunc("POST "+p, s.handle(ir.FormatOpenAIResponses))
	}
	for _, p := range []string{"/v1/messages", "/messages"} {
		mux.HandleFunc("POST "+p, s.handle(ir.FormatClaude))
	}
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /models", s.handleModels)
	mux.HandleFunc("POST /v1beta/models/{name}", s.handleGemini)
	mux.HandleFunc("POST /v1/models/{name}", s.handleGemini)
	return mux
}

func (s *Server) handle(format ir.Format) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		s.serve(w, r, inbound{format: format})
	}
}

func (s *Server) handleGemini(w http.ResponseWriter, r *http.Request) {
	model, action, err := splitGemini(r.PathValue("name"))
	if err != nil {
		writeErr(w, ir.FormatGemini, http.StatusNotFound, "not_found_error", err.Error())
		return
	}
	switch action {
	case "generateContent", "streamGenerateContent":
	default:
		writeErr(w, ir.FormatGemini, http.StatusNotFound, "not_found_error", "unknown gemini action "+action)
		return
	}
	s.serve(w, r, inbound{
		format:      ir.FormatGemini,
		model:       model,
		forceStream: action == "streamGenerateContent" || r.URL.Query().Get("alt") == "sse",
	})
}

type inbound struct {
	format      ir.Format
	model       string
	forceStream bool
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request, in inbound) {
	start := time.Now()
	status := http.StatusOK
	upName := ""
	var model string
	defer func() {
		slog.Info("relay", "status", status, "model", model, "upstream", upName, "from", in.format, "path", r.URL.Path, "ms", time.Since(start).Milliseconds())
	}()

	body, err := readBody(w, r)
	if err != nil {
		status = errStatus(err)
		writeErr(w, in.format, status, ir.ErrorTypeForStatus(status), err.Error())
		return
	}
	if !s.cfg.authorized(clientKey(r)) {
		status = writeErr(w, in.format, http.StatusUnauthorized, "authentication_error", "invalid or missing API key")
		return
	}

	var stream bool
	model, stream = peekRequest(body)
	if in.model != "" {
		model = in.model
	}
	if in.forceStream {
		stream = true
	}
	if model == "" {
		status = writeErr(w, in.format, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}
	route, up, ok := s.cfg.match(model)
	if !ok {
		status = writeErr(w, in.format, http.StatusNotFound, "not_found_error", "no upstream for model "+model)
		return
	}
	upName = up.Name
	if up.Format == in.format {
		status = s.passthrough(w, r, in.format, up, route, body, model, stream)
		return
	}

	src, err := convert.Get(in.format)
	if err != nil {
		status = writeErr(w, in.format, http.StatusNotFound, "not_found_error", err.Error())
		return
	}
	req, err := src.DecodeRequest(body)
	if err != nil {
		status = writeErr(w, in.format, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	req.Model, req.Stream = model, stream
	clientModel := model
	if route.UpstreamModel != "" {
		req.Model = route.UpstreamModel
	}

	dst, err := convert.Get(up.Format)
	if err != nil {
		status = writeErr(w, in.format, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	encoded, err := dst.EncodeRequest(req)
	if err != nil {
		status = writeErr(w, in.format, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if err := s.cfg.Loss.Check(req.Diags); err != nil {
		status = writeErr(w, in.format, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if len(req.Diags) > 0 {
		slog.Info("conversion", "model", clientModel, "diags", req.Diags.String())
	}

	view := *req
	view.Model = clientModel

	endpoint, err := upstreamURL(up, req.Model, req.Stream)
	if err != nil {
		status = writeErr(w, in.format, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	ctx := r.Context()
	var cancel context.CancelFunc
	if s.cfg.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, s.cfg.Timeout)
		defer cancel()
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(encoded))
	if err != nil {
		status = writeErr(w, in.format, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	httpReq.Header = upstreamHeaders(r, up, req.Stream)

	resp, err := s.http.Do(httpReq)
	if err != nil {
		slog.Error("upstream", "upstream", up.Name, "err", err)
		status = writeErr(w, in.format, http.StatusBadGateway, "api_error", "upstream request failed")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		status = writeAPI(w, in.format, dst.DecodeError(resp.StatusCode, b))
		return
	}

	sseUp := isEventStream(resp.Header) || (req.Stream && !isJSON(resp.Header))
	if req.Stream {
		status = s.writeStream(w, in.format, up.Format, sseUp, resp.Body, &view)
		return
	}
	if sseUp {
		parsed, apiErr, err := Collect(up.Format, resp.Body)
		if err != nil {
			status = writeErr(w, in.format, http.StatusBadGateway, "api_error", err.Error())
			return
		}
		if apiErr != nil {
			status = writeAPI(w, in.format, apiErr)
			return
		}
		if clientModel != "" {
			parsed.Model = clientModel
		}
		out, err := src.EncodeResponse(parsed, &view)
		if err != nil {
			status = writeErr(w, in.format, http.StatusBadGateway, "api_error", err.Error())
			return
		}
		status = writeJSON(w, http.StatusOK, out)
		return
	}

	b, err := io.ReadAll(io.LimitReader(resp.Body, maxBody))
	if err != nil {
		status = writeErr(w, in.format, http.StatusBadGateway, "api_error", "reading upstream response")
		return
	}
	parsed, err := dst.DecodeResponse(b)
	if err != nil {
		status = writeErr(w, in.format, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	if clientModel != "" {
		parsed.Model = clientModel
	}
	out, err := src.EncodeResponse(parsed, &view)
	if err != nil {
		status = writeErr(w, in.format, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	status = writeJSON(w, http.StatusOK, out)
}

func (s *Server) writeStream(w http.ResponseWriter, to, from ir.Format, sseUp bool, body io.Reader, view *ir.Request) int {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flush := func() {}
	if f, ok := w.(http.Flusher); ok {
		flush = f.Flush
	}
	var err error
	if sseUp {
		err = Stream(from, to, body, w, flush, view)
	} else {
		b, rerr := io.ReadAll(io.LimitReader(body, maxBody))
		if rerr != nil {
			err = rerr
		} else {
			parsed, derr := convert.MustGet(from).DecodeResponse(b)
			if derr != nil {
				err = derr
			} else {
				if view != nil && view.Model != "" {
					parsed.Model = view.Model
				}
				err = writeEvents(to, eventsFromResponse(parsed), w, flush, view)
			}
		}
	}
	if err != nil {
		slog.Error("stream", "err", err)
	}
	return http.StatusOK
}

func splitGemini(name string) (model, action string, err error) {
	name, err = url.PathUnescape(name)
	if err != nil {
		return "", "", err
	}
	i := strings.LastIndex(name, ":")
	if i <= 0 || i == len(name)-1 {
		return "", "", errors.New("gemini path must be models/{model}:{action}")
	}
	return name[:i], name[i+1:], nil
}

func readBody(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBody)
	b, err := io.ReadAll(r.Body)
	if err != nil {
		var max *http.MaxBytesError
		if errors.As(err, &max) {
			return nil, errTooLarge
		}
		return nil, fmt.Errorf("read body: %w", err)
	}
	return b, nil
}

var errTooLarge = errors.New("request body is too large")

func errStatus(err error) int {
	if errors.Is(err, errTooLarge) {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusBadRequest
}

func isEventStream(h http.Header) bool {
	return strings.Contains(strings.ToLower(h.Get("Content-Type")), "text/event-stream")
}

func isJSON(h http.Header) bool {
	return strings.Contains(strings.ToLower(h.Get("Content-Type")), "json")
}

func writeErr(w http.ResponseWriter, format ir.Format, status int, typ, msg string) int {
	return writeAPI(w, format, &ir.APIError{Status: status, Type: typ, Message: msg})
}

func writeAPI(w http.ResponseWriter, format ir.Format, e *ir.APIError) int {
	if e == nil {
		e = &ir.APIError{Status: http.StatusInternalServerError, Type: "api_error", Message: "unknown error"}
	}
	status := e.Status
	if status == 0 {
		status = http.StatusInternalServerError
	}
	body := []byte(`{"error":{"message":"internal error"}}`)
	if c, err := convert.Get(format); err == nil {
		body = c.EncodeError(e)
	}
	return writeJSON(w, status, body)
}

func writeJSON(w http.ResponseWriter, status int, body []byte) int {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
	return status
}

func errf(format string, args ...any) error { return fmt.Errorf(format, args...) }
