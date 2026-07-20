// examples/fallback/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	openaioption "github.com/openai/openai-go/v2/option"

	"github.com/habibiramadhan-dev/aisdk-go"
	anthropicprovider "github.com/habibiramadhan-dev/aisdk-go/providers/anthropic"
	geminiprovider "github.com/habibiramadhan-dev/aisdk-go/providers/gemini"
	openaiprovider "github.com/habibiramadhan-dev/aisdk-go/providers/openai"
)

func main() {
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	if anthropicKey == "" {
		log.Fatal("set ANTHROPIC_API_KEY to run this example")
	}
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		log.Fatal("set OPENAI_API_KEY to run this example")
	}
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if geminiKey == "" {
		log.Fatal("set GEMINI_API_KEY to run this example")
	}

	ctx := context.Background()

	// Fallback has its own retry/backoff policy (WithMaxRetries/WithBackoff,
	// configured below), so each provider SDK's own built-in retry is
	// disabled here via option.WithMaxRetries(0). Both the Anthropic and
	// OpenAI SDKs default to 2 automatic retries internally — leaving that
	// enabled would silently retry the same transient failure inside the
	// SDK itself before Fallback ever saw an error, effectively
	// double-retrying (and double-waiting) on top of Fallback's own
	// retry/backoff/fallback timing.
	anthropicProv := anthropicprovider.New(anthropicKey, anthropicoption.WithMaxRetries(0))
	anthropicModel := anthropicProv.Model("claude-sonnet-5")

	openaiProv := openaiprovider.New(openaiKey, openaioption.WithMaxRetries(0))
	openaiModel := openaiProv.Model("gpt-4o")

	// Gemini's SDK has no built-in retry loop for Generate/Stream calls at
	// all, so there's no equivalent option to disable here — unlike the two
	// providers above, nothing needs to be turned off for Gemini.
	geminiProv, err := geminiprovider.New(ctx, geminiKey)
	if err != nil {
		log.Fatalf("gemini: constructing provider failed: %v", err)
	}
	geminiModel := geminiProv.Model("gemini-2.0-flash")

	// Fallback tries anthropicModel first. A retryable failure on it is
	// retried (with backoff) up to WithMaxRetries times before Fallback
	// moves on to openaiModel, then geminiModel, applying the same
	// retry/backoff policy at each step. Everything below this point uses
	// the wrapped model exactly like any single aisdk.Model — if the first
	// model in the chain fails with a retryable error, Fallback transparently
	// retries and/or falls back to the next provider; the caller's code
	// doesn't change at all compared to calling a single provider directly.
	model := aisdk.Fallback(
		[]aisdk.Model{anthropicModel, openaiModel, geminiModel},
		aisdk.WithMaxRetries(2),
	)

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
}
