# aisdk-go

A single Go interface for calling Anthropic, OpenAI, and Gemini — one
request/response shape, one streaming API, one tool-calling contract, one
structured-output API, instead of three different provider SDKs with three
different shapes. It's for Go developers building LLM-backed applications
who want to write against one `aisdk.Model` interface and swap providers
(or add retry/fallback across them) without rewriting call sites. It's
modeled on the developer experience of Vercel's AI SDK, translated into
idiomatic Go.

What you get beyond calling each provider's official SDK directly: a
unified `Generate`/`Stream` request and response shape across all three
providers, a unified tool-calling contract, generic structured-output
extraction (`GenerateStructured[T]`), a provider-agnostic retry/fallback
decorator (`Fallback`), and an OpenTelemetry observability decorator
(`otel.Wrap`) — all implemented today, not on a roadmap, and all
conformance-tested against fake HTTP transports so behavior is verified
identically across providers.

## Status

Feature-complete and pre-1.0: `Generate`, `Stream`, tool-calling,
structured output, fallback/retry, and OpenTelemetry observability are all
implemented and conformance-tested for Anthropic, OpenAI, and Gemini alike.
No `v1.0.0` tag has been cut yet — treat the API as stable in shape but not
yet formally versioned.

## Requirements

- Go 1.26.5 or newer (see `go.mod`)

```bash
go get github.com/habibiramadhan-dev/aisdk-go
```

## Quickstart

Get an API key at https://console.anthropic.com, then:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
go run github.com/habibiramadhan-dev/aisdk-go/examples/basic-chat
```

```go
import anthropicprovider "github.com/habibiramadhan-dev/aisdk-go/providers/anthropic"

provider := anthropicprovider.New(os.Getenv("ANTHROPIC_API_KEY"))
model := provider.Model("claude-sonnet-5")

answer, err := aisdk.Ask(context.Background(), model, "hello")
```

## Providers

Every provider is constructed with a `New(...)` on its package, then bound
to a specific model name via `.Model(name)`, returning an `aisdk.Model` —
the same interface regardless of provider.

| Provider | Import path | Env var | Example model name |
|---|---|---|---|
| Anthropic | `github.com/habibiramadhan-dev/aisdk-go/providers/anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5` |
| OpenAI | `github.com/habibiramadhan-dev/aisdk-go/providers/openai` | `OPENAI_API_KEY` | `gpt-4o` |
| Gemini | `github.com/habibiramadhan-dev/aisdk-go/providers/gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` |

Note: unlike the Anthropic and OpenAI adapters, `gemini.New` takes a
`context.Context` and returns an error — its constructor is fallible.

## Features

- **Generate** — single-shot request/response, unified across providers. See `examples/basic-chat`.
- **Stream** — token-level streaming via a `<-chan StreamEvent`. See `examples/basic-chat`.
- **Tool-calling** — declare `aisdk.Tool{Name, Description, Parameters}` on a request, inspect `ToolCall`s in the response. See `examples/tool-calling`.
- **Structured output** — `aisdk.GenerateStructured[T]` reflects a Go type into a JSON Schema and unmarshals the constrained response into it. See `examples/structured-output`.
- **Fallback / retry** — `aisdk.Fallback(models, opts...)` wraps a chain of models with configurable retry, backoff, and a time budget. See `examples/fallback`.
- **Observability** — `otel.Wrap(model, opts...)` decorates any `aisdk.Model` with OpenTelemetry GenAI semantic-convention spans. See `examples/otel-tracing`.

## Coming from Vercel AI SDK

| Vercel AI SDK | aisdk-go | Notes |
|---|---|---|
| `generateText()` | `model.Generate(ctx, req)`, or `aisdk.Ask(ctx, model, prompt)` for the simple case | |
| `streamText()` | `model.Stream(ctx, req)` | returns `<-chan aisdk.StreamEvent` |
| `generateObject()` | `aisdk.GenerateStructured[T](ctx, model, req)` | `T` is any Go struct; its JSON Schema is derived via reflection |
| `tool()` | `aisdk.Tool{Name, Description, Parameters}` in `GenerateRequest.Tools` | `Parameters` is a raw JSON Schema (`json.RawMessage`) |
| `wrapLanguageModel()` | `aisdk.Fallback(...)` / `otel.Wrap(...)` | no dedicated middleware type — just Go interface composition, since every decorator implements the same `aisdk.Model` interface |

**No automatic multi-step tool-calling loop.** Unlike some higher-level SDKs,
v1 of aisdk-go does not run the tool-call loop for you. The caller drives it
manually: call `Generate`, check `resp.FinishReason == aisdk.FinishReasonToolCalls`,
pull the `ToolCall` out of `resp.Message.Parts`, execute the tool yourself,
append the result as a `RoleTool` message, and call `Generate` again. See
`examples/tool-calling/main.go` for the full reference implementation of
this pattern.

## Examples

All examples live under `examples/` and are runnable with `go run`:

- [`examples/basic-chat`](examples/basic-chat) — `Ask`, `Generate`, and `Stream` against Anthropic.
- [`examples/tool-calling`](examples/tool-calling) — the manual two-call tool-calling loop.
- [`examples/structured-output`](examples/structured-output) — `GenerateStructured[T]` extracting a typed struct from a model response.
- [`examples/fallback`](examples/fallback) — chaining Anthropic → OpenAI → Gemini with retry/backoff via `aisdk.Fallback`.
- [`examples/otel-tracing`](examples/otel-tracing) — wrapping a model with `otel.Wrap` and exporting spans via stdout.

## Security

See [`SECURITY.md`](SECURITY.md) for the vulnerability reporting process.

One fact worth calling out directly here: `ToolCall.Arguments` is
untrusted, model-generated JSON. The SDK never validates, sandboxes, or
executes it — that responsibility is entirely the caller's, as shown in
`examples/tool-calling/main.go`.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to get set up and submit changes.

## License

MIT — see [`LICENSE`](LICENSE).
