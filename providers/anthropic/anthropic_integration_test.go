//go:build integration

package anthropic_test

import (
	"context"
	"os"
	"testing"

	"github.com/habibiramadhan-dev/aisdk-go"
	anthropicprovider "github.com/habibiramadhan-dev/aisdk-go/providers/anthropic"
)

// TestIntegration_Generate_RealAPI hits the real Anthropic API. It's only
// compiled when the "integration" build tag is passed (go test
// -tags=integration ./...) — normal `go test ./...` (including CI's default
// job) never sees this file at all, so it costs nothing when API keys
// aren't available. Skips cleanly (not a failure) when ANTHROPIC_API_KEY
// isn't set, so `go test -tags=integration ./...` run locally without any
// keys configured still passes, just skipping this specific test.
func TestIntegration_Generate_RealAPI(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider := anthropicprovider.New(apiKey)
	model := provider.Model("claude-sonnet-5")

	resp, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("Say hello in one word.")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	var text string
	for _, part := range resp.Message.Parts {
		if part.Type == aisdk.ContentPartTypeText {
			text += part.Text
		}
	}
	if text == "" {
		t.Error("Generate returned no text content from the real API")
	}
	t.Logf("real API response: %q, finish reason: %s, usage: %+v", text, resp.FinishReason, resp.Usage)
}
