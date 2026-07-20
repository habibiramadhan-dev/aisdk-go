// examples/tool-calling/main.go
package main

import (
	"context"
	"encoding/json"
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

	getWeather := aisdk.Tool{
		Name:        "get_weather",
		Description: "Gets the current weather for a location",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
	}

	userMessage := aisdk.Message{
		Role:  aisdk.RoleUser,
		Parts: []aisdk.ContentPart{aisdk.TextPart("What's the weather in Paris?")},
	}

	// First call: give the model the tool declaration and let it decide
	// whether to call it.
	resp, err := model.Generate(ctx, aisdk.GenerateRequest{
		Messages:  []aisdk.Message{userMessage},
		Tools:     []aisdk.Tool{getWeather},
		MaxTokens: 256,
	})
	if err != nil {
		log.Fatalf("Generate failed: %v", err)
	}

	if resp.FinishReason != aisdk.FinishReasonToolCalls {
		// The model answered directly without calling the tool.
		for _, part := range resp.Message.Parts {
			if part.Type == aisdk.ContentPartTypeText {
				fmt.Println("Generate():", part.Text)
			}
		}
		return
	}

	var toolCall *aisdk.ToolCall
	for _, part := range resp.Message.Parts {
		if part.Type == aisdk.ContentPartTypeToolCall {
			toolCall = part.ToolCall
		}
	}
	if toolCall == nil {
		log.Fatal("FinishReason was ToolCalls but resp.Message.Parts has no tool call part")
	}
	fmt.Printf("Model called tool %q with arguments: %s\n", toolCall.Name, toolCall.Arguments)

	// toolCall.Arguments is untrusted, model-generated JSON (see design.md
	// §8) — the SDK never validates or executes it. A real caller MUST
	// validate/sandbox these arguments (schema-check them, allowlist the
	// tool name, bound any file/network access, etc.) before acting on
	// them. This example skips that step and hardcodes a fake result
	// purely for demonstration purposes.
	fakeResult := `{"temperature_celsius": 18, "condition": "cloudy"}`

	// Second call: echo the tool call back as an assistant turn, then
	// supply the (fake) result as a RoleTool message, and let the model
	// produce its final answer.
	followUp := aisdk.GenerateRequest{
		Messages: []aisdk.Message{
			userMessage,
			{
				Role: aisdk.RoleAssistant,
				Parts: []aisdk.ContentPart{{
					Type:     aisdk.ContentPartTypeToolCall,
					ToolCall: toolCall,
				}},
			},
			{
				Role: aisdk.RoleTool,
				Parts: []aisdk.ContentPart{{
					Type: aisdk.ContentPartTypeToolResult,
					ToolResult: &aisdk.ToolResult{
						ToolCallID: toolCall.ID,
						Content:    fakeResult,
					},
				}},
			},
		},
		Tools:     []aisdk.Tool{getWeather},
		MaxTokens: 256,
	}

	finalResp, err := model.Generate(ctx, followUp)
	if err != nil {
		log.Fatalf("follow-up Generate failed: %v", err)
	}
	for _, part := range finalResp.Message.Parts {
		if part.Type == aisdk.ContentPartTypeText {
			fmt.Println("Final answer:", part.Text)
		}
	}
}
