# llm-one-api

HTTP gateway that lets a client speak one LLM wire format and an upstream speak another. A request is decoded into a neutral intermediate representation and encoded again for the provider that will actually run it.

Supported formats:

| Format | Config value | Client paths |
| --- | --- | --- |
| OpenAI Chat Completions | `openai_chat` | `POST /v1/chat/completions`, `POST /chat/completions` |
| OpenAI Responses | `openai_responses` | `POST /v1/responses`, `POST /responses` |
| Claude Messages | `claude` | `POST /v1/messages`, `POST /messages` |
| Gemini generateContent | `gemini` | `POST /v1beta/models/{model}:generateContent`, `POST /v1beta/models/{model}:streamGenerateContent`, and the same paths under `/v1/models/` |

`GET /healthz` returns `ok`.

Streaming is forwarded event by event. If the client asked for a stream and the upstream returned one JSON body, the relay turns that body into server-sent events. The response uses the model name the client sent. `upstream_model` is only what the provider sees.

## Run

Go 1.26 or newer.

```powershell
copy config.example.yaml config.yaml
go run ./cmd/llm-one-api -config config.yaml
```

The server listens on `:8080` unless `listen` is set. `config.yaml` is gitignored because it holds API keys.

```powershell
go test ./...
```

## Configuration

`config.example.yaml` is a complete example. `${ENV_VAR}` is expanded in string fields.

```yaml
listen: ":8080"
loss: allow # allow | safe | strict
timeout: 10m

upstreams:
  - name: openai
    format: openai_chat
    base_url: https://api.openai.com
    api_key: ${OPENAI_API_KEY}

routes:
  - model: claude-*
    upstream: anthropic
    upstream_model: claude-sonnet-4-5
  - model: "*"
    upstream: openai
```

`base_url` may be an origin (`https://api.openai.com`) or already include the version prefix (`https://api.openai.com/v1`). The version segment is not added twice.

Routes are first match wins. `*` is the only glob metacharacter, so `claude-*` and `*-mini` both work. With a single upstream and no routes, every model uses that upstream.

`loss` controls conversion that cannot be represented exactly:

- `allow` sends the request anyway
- `safe` rejects error diagnostics
- `strict` rejects warnings and errors

`timeout` bounds the whole upstream call, including a stream.

## Credentials

An upstream `api_key` is sent as `Authorization: Bearer` for OpenAI, `x-api-key` for Claude, and `x-goog-api-key` for Gemini. Claude requests also get `anthropic-version: 2023-06-01` when the client did not send one. Extra provider headers can be set on the upstream with `headers`.

If `api_key` is empty, the relay reuses the client credential from `Authorization: Bearer`, `x-api-key`, `x-goog-api-key`, or the `key` query parameter, and translates it into the upstream's header.

## What is preserved

Tools, tool results, images, and thinking blocks are converted when the destination format has a place for them. Thinking signatures and Gemini thought signatures are replayed only to the provider that issued them. A signature that has to travel through another client format is wrapped and unpacked on the way back. Hosted tools that exist on only one provider, such as Claude server tools, are dropped and recorded as a warning.
