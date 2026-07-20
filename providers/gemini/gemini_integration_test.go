//go:build integration

package gemini_test

import (
	"context"
	"os"
	"testing"

	"github.com/habibiramadhan-dev/aisdk-go"
	geminiprovider "github.com/habibiramadhan-dev/aisdk-go/providers/gemini"
)

// TestIntegration_Generate_RealAPI hits the real Gemini API. It's only
// compiled when the "integration" build tag is passed (go test
// -tags=integration ./...) — normal `go test ./...` (including CI's default
// job) never sees this file at all, so it costs nothing when API keys
// aren't available. Skips cleanly (not a failure) when GEMINI_API_KEY isn't
// set — checked BEFORE attempting construction, since gemini's New is
// fallible and shouldn't be called with an empty key — so `go test
// -tags=integration ./...` run locally without any keys configured still
// passes, just skipping this specific test.
func TestIntegration_Generate_RealAPI(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	provider, err := geminiprovider.New(context.Background(), apiKey)
	if err != nil {
		t.Fatalf("failed to construct gemini provider: %v", err)
	}
	model := provider.Model("gemini-2.0-flash")

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
