// examples/basic-chat/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/habibiramadhan-dev/aisdk-go"
	anthropicprovider "github.com/habibiramadhan-dev/aisdk-go/providers/anthropic"
)

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("set ANTHROPIC_API_KEY to run this example")
	}

	provider := anthropicprovider.New(apiKey)
	model := provider.Model("claude-sonnet-5")
	ctx := context.Background()

	// Simple path.
	answer, err := aisdk.Ask(ctx, model, "In one sentence, what is Go good at?")
	if err != nil {
		log.Fatalf("Ask failed: %v", err)
	}
	fmt.Println("Ask():", answer)

	// Full Generate API.
	resp, err := model.Generate(ctx, aisdk.GenerateRequest{
		System:    "You are a terse assistant. Answer in five words or fewer.",
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("What is Go?")}}},
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

	// Streaming.
	stream, err := model.Stream(ctx, aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("Count from 1 to 5.")}}},
		MaxTokens: 64,
	})
	if err != nil {
		log.Fatalf("Stream failed: %v", err)
	}
	fmt.Print("Stream(): ")
	var streamErr error
	for event := range stream {
		switch event.Type {
		case aisdk.StreamEventTypeTextDelta:
			fmt.Print(event.Delta)
		case aisdk.StreamEventTypeFinish:
			fmt.Printf("\n[finished: %s, %d input / %d output tokens]\n", event.FinishReason, event.Usage.InputTokens, event.Usage.OutputTokens)
		case aisdk.StreamEventTypeError:
			streamErr = event.Err
		}
	}
	if streamErr != nil {
		log.Fatalf("stream error: %v", streamErr)
	}
}
