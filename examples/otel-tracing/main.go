// examples/otel-tracing/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/habibiramadhan-dev/aisdk-go"
	aisdkotel "github.com/habibiramadhan-dev/aisdk-go/otel"
	anthropicprovider "github.com/habibiramadhan-dev/aisdk-go/providers/anthropic"
)

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("set ANTHROPIC_API_KEY to run this example")
	}

	ctx := context.Background()

	// stdouttrace prints readable span JSON to stdout — no Phoenix/collector
	// needed to see otel.Wrap producing real spans. WithBatcher mirrors how a
	// production exporter is normally configured (batched, async), as opposed
	// to the test suite's WithSyncer (synchronous, no flush needed) — that's
	// why this example explicitly calls tp.Shutdown(ctx) before exiting below.
	exporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		log.Fatalf("failed to create stdout exporter: %v", err)
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter))
	otel.SetTracerProvider(tp)

	// To send traces to a real Phoenix instance instead of stdout, replace the
	// stdouttrace exporter above with:
	//
	//   import "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	//   exporter, err := otlptracehttp.New(ctx, otlptracehttp.WithEndpointURL("http://localhost:6006/v1/traces"))
	//
	// This is the setup needed to satisfy design.md §10 Fase 7's stated exit
	// criterion ("traces show up correctly in a real self-hosted Phoenix
	// instance for a live call") — running it against one, with a real
	// ANTHROPIC_API_KEY, is a manual verification step for whoever runs this
	// example, the same way every other example in this repo needs real
	// credentials to actually execute (this repo's automated tests never make
	// real network calls to any provider or tracing backend).

	provider := anthropicprovider.New(apiKey)
	baseModel := provider.Model("claude-sonnet-5")

	// WithCaptureContent is opt-in and OFF by default per design.md §8 —
	// consumer applications may put PII or secrets in prompts they never
	// intended to export to a tracing backend, so recording full
	// prompt/completion text as span attributes must be an explicit choice,
	// not a default. It's turned on here only to make the example's printed
	// spans more interesting to inspect.
	model := aisdkotel.Wrap(baseModel, aisdkotel.WithCaptureContent())

	resp, err := model.Generate(ctx, aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("In one sentence, what is Go good at?")}}},
		MaxTokens: 64,
	})
	if err != nil {
		log.Fatalf("Generate failed: %v", err)
	}
	for _, part := range resp.Message.Parts {
		if part.Type == aisdk.ContentPartTypeText {
			fmt.Println("Generate():", part.Text)
		}
	}
	fmt.Printf("Usage: %d input tokens, %d output tokens\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)

	// Required to flush the stdout exporter's batched span — WithBatcher (the
	// production-shaped configuration used above) buffers spans and exports
	// them asynchronously, unlike the test suite's WithSyncer, which exports
	// synchronously as each span ends. Without this Shutdown call, the
	// process could exit before the span is ever written to stdout.
	if err := tp.Shutdown(ctx); err != nil {
		log.Fatalf("tracer provider shutdown failed: %v", err)
	}
}
