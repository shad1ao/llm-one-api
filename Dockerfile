FROM golang:1.26.6 AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/llm-one-api ./cmd/llm-one-api

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/llm-one-api /llm-one-api
COPY config.example.yaml /config.yaml

EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/llm-one-api"]
CMD ["-config", "/config.yaml"]
