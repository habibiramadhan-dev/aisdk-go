// aisdktest/conformance.go
package aisdktest

import (
	"context"
	"encoding/json"
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

	t.Run("Generate_ReturnsToolCall", func(t *testing.T) {
		model := newModel(t)

		resp, err := model.Generate(context.Background(), aisdk.GenerateRequest{
			Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("What's the weather in Paris?")}}},
			Tools: []aisdk.Tool{{
				Name:        "get_weather",
				Description: "Gets the current weather for a location",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
			}},
			MaxTokens: 64,
		})
		if err != nil {
			t.Fatalf("Generate returned error: %v", err)
		}

		var toolCall *aisdk.ToolCall
		for _, part := range resp.Message.Parts {
			if part.Type == aisdk.ContentPartTypeToolCall {
				toolCall = part.ToolCall
			}
		}
		if toolCall == nil {
			t.Fatal("Generate returned no tool call")
		}
		if toolCall.Name == "" {
			t.Error("tool call has an empty Name")
		}
		if toolCall.ID == "" {
			t.Error("tool call has an empty ID")
		}
		if len(toolCall.Arguments) == 0 {
			t.Error("tool call has empty Arguments")
		}
		if resp.FinishReason != aisdk.FinishReasonToolCalls {
			t.Errorf("resp.FinishReason = %q, want %q", resp.FinishReason, aisdk.FinishReasonToolCalls)
		}
	})

	t.Run("GenerateStructured_ReturnsTypedResult", func(t *testing.T) {
		model := newModel(t)

		type weatherReport struct {
			City         string  `json:"city"`
			TemperatureC float64 `json:"temperature_c"`
		}

		result, err := aisdk.GenerateStructured[weatherReport](context.Background(), model, aisdk.GenerateRequest{
			Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("What's the weather in Paris?")}}},
			MaxTokens: 64,
		})
		if err != nil {
			t.Fatalf("GenerateStructured returned error: %v", err)
		}

		if result.City == "" {
			t.Error("result.City is empty")
		}
		if result.TemperatureC == 0 {
			t.Error("result.TemperatureC is zero — likely didn't parse correctly")
		}
	})
}
