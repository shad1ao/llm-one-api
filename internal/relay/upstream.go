package relay

import (
	"net/http"
	"net/url"
	"strings"

	"llm-one-api/internal/convert/ir"
)

// upstreamURL builds the provider endpoint for a converted request.
// base may be an origin (https://api.openai.com) or already include the
// version prefix (https://api.openai.com/v1); the version is not doubled.
func upstreamURL(up Upstream, model string, stream bool) (string, error) {
	var path string
	switch up.Format {
	case ir.FormatOpenAIChat:
		path = "/v1/chat/completions"
	case ir.FormatOpenAIResponses:
		path = "/v1/responses"
	case ir.FormatClaude:
		path = "/v1/messages"
	case ir.FormatGemini:
		action := "generateContent"
		if stream {
			action = "streamGenerateContent"
		}
		model = strings.TrimPrefix(model, "models/")
		path = "/v1beta/models/" + url.PathEscape(model) + ":" + action
		if stream {
			path += "?alt=sse"
		}
	default:
		return "", errf("unsupported upstream format %q", up.Format)
	}
	return joinBase(up.BaseURL, path), nil
}

func joinBase(base, path string) string {
	base = strings.TrimRight(base, "/")
	switch {
	case strings.HasSuffix(base, "/v1") && strings.HasPrefix(path, "/v1/"):
		return base + strings.TrimPrefix(path, "/v1")
	case strings.HasSuffix(base, "/v1beta") && strings.HasPrefix(path, "/v1beta/"):
		return base + strings.TrimPrefix(path, "/v1beta")
	default:
		return base + path
	}
}

func upstreamHeaders(r *http.Request, up Upstream, stream bool) http.Header {
	h := make(http.Header)
	h.Set("Content-Type", "application/json")
	if stream {
		h.Set("Accept", "text/event-stream")
	} else {
		h.Set("Accept", "application/json")
	}
	key := up.APIKey
	if key == "" {
		key = clientKey(r)
	}
	if key != "" {
		switch up.Format {
		case ir.FormatClaude:
			h.Set("x-api-key", key)
		case ir.FormatGemini:
			h.Set("x-goog-api-key", key)
		default:
			h.Set("Authorization", "Bearer "+key)
		}
	}
	if up.Format == ir.FormatClaude {
		ver := r.Header.Get("anthropic-version")
		if ver == "" {
			ver = "2023-06-01"
		}
		h.Set("anthropic-version", ver)
		if beta := r.Header.Get("anthropic-beta"); beta != "" {
			h.Set("anthropic-beta", beta)
		}
	}
	for k, v := range up.Headers {
		h.Set(k, v)
	}
	return h
}

func clientKey(r *http.Request) string {
	if r == nil {
		return ""
	}
	if k := strings.TrimSpace(r.Header.Get("x-api-key")); k != "" {
		return k
	}
	if k := strings.TrimSpace(r.Header.Get("x-goog-api-key")); k != "" {
		return k
	}
	if k := strings.TrimSpace(r.URL.Query().Get("key")); k != "" {
		return k
	}
	auth := r.Header.Get("Authorization")
	if rest, ok := strings.CutPrefix(auth, "Bearer "); ok {
		return strings.TrimSpace(rest)
	}
	return ""
}
