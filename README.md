# aisdk-go

A single Go interface for calling Anthropic, OpenAI, and Gemini — one request/
response shape and one streaming/tool-calling/structured-output API instead of
three different provider SDKs. Modeled on Vercel AI SDK's developer experience.

**Status:** early and under active development. Anthropic works today
(non-streaming only). OpenAI and Gemini support, streaming, tool-calling, and
structured output are all planned but not built yet.

**Requires Go 1.22 or newer.**

## Quickstart

Get an API key at https://console.anthropic.com, then:

```bash
go get github.com/habibiramadhan-dev/aisdk-go
export ANTHROPIC_API_KEY=sk-ant-...
go run github.com/habibiramadhan-dev/aisdk-go/examples/basic-chat
```

```go
import anthropicprovider "github.com/habibiramadhan-dev/aisdk-go/providers/anthropic"

provider := anthropicprovider.New(os.Getenv("ANTHROPIC_API_KEY"))
model := provider.Model("claude-sonnet-5")

answer, err := aisdk.Ask(context.Background(), model, "hello")
```
