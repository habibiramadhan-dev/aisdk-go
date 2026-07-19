// aisdktest/conformance.go
package aisdktest

import (
	"context"
	"testing"

	"github.com/habibiramadhan-dev/aisdk-go"
)

// RunConformanceSuite runs the shared contract tests every provider adapter
// must pass. newModel is called once per sub-test and must return a Model
// wired up however the adapter needs (fake transport, real network, etc.) —
// RunConformanceSuite itself never makes network assumptions.
func RunConformanceSuite(t *testing.T, newModel func(t *testing.T) aisdk.Model) {
	t.Run("Generate_ReturnsNonEmptyText", func(t *testing.T) {
		model := newModel(t)

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
			t.Error("Generate returned no text content")
		}
	})

	t.Run("Stream_EndsWithFinishEvent", func(t *testing.T) {
		model := newModel(t)

		stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{
			Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("Say hello in one word.")}}},
			MaxTokens: 64,
		})
		if err != nil {
			t.Fatalf("Stream returned error: %v", err)
		}

		var sawFinish bool
		var lastType aisdk.StreamEventType
		for event := range stream {
			if event.Type == aisdk.StreamEventTypeError {
				t.Fatalf("Stream emitted an error event: %v", event.Err)
			}
			if event.Type == aisdk.StreamEventTypeFinish {
				sawFinish = true
			}
			lastType = event.Type
		}
		if !sawFinish {
			t.Error("Stream never emitted a Finish event")
		}
		if lastType != aisdk.StreamEventTypeFinish {
			t.Errorf("last event type = %q, want the stream to end with Finish", lastType)
		}
	})
}
