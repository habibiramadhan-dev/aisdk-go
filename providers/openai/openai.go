package openai

import (
	"context"

	openaisdk "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/habibiramadhan-dev/aisdk-go"
)

// Provider constructs OpenAI-backed aisdk.Model values.
type Provider struct {
	client openaisdk.Client
}

// New constructs a Provider. Extra option.RequestOption values (e.g.
// option.WithBaseURL for tests) are applied after the API key.
func New(apiKey string, opts ...option.RequestOption) *Provider {
	allOpts := append([]option.RequestOption{option.WithAPIKey(apiKey)}, opts...)
	return &Provider{client: openaisdk.NewClient(allOpts...)}
}

// Model returns an aisdk.Model bound to the given OpenAI model name
// (e.g. "gpt-4o").
func (p *Provider) Model(name string) aisdk.Model {
	return &model{client: p.client, modelName: name}
}

type model struct {
	client    openaisdk.Client
	modelName string
}

func (m *model) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	params := toChatCompletionParams(m.modelName, req)

	resp, err := m.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return aisdk.GenerateResponse{}, mapError(err)
	}

	return toGenerateResponse(resp)
}

// Stream's returned channel is closed once the response ends, errors, or ctx
// is cancelled. Callers that stop ranging over the channel before it closes
// on its own MUST cancel ctx — otherwise the background goroutine blocks
// forever trying to send the next event to a channel nobody is reading.
func (m *model) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	params := toChatCompletionParams(m.modelName, req)
	params.StreamOptions = openaisdk.ChatCompletionStreamOptionsParam{
		IncludeUsage: openaisdk.Bool(true),
	}
	sdkStream := m.client.Chat.Completions.NewStreaming(ctx, params)

	events := make(chan aisdk.StreamEvent)
	go func() {
		defer close(events)
		defer sdkStream.Close()

		var finishReason aisdk.FinishReason
		var usage aisdk.Usage
		toolCallIDs := make(map[int64]string)
		toolCallNames := make(map[int64]string)

		send := func(e aisdk.StreamEvent) bool {
			select {
			case events <- e:
				return true
			case <-ctx.Done():
				return false
			}
		}

		for sdkStream.Next() {
			chunk := sdkStream.Current()

			if se, ok := toStreamEvent(chunk); ok {
				if !send(se) {
					return
				}
			}

			if len(chunk.Choices) > 0 {
				for _, deltaTool := range chunk.Choices[0].Delta.ToolCalls {
					if deltaTool.ID != "" {
						toolCallIDs[deltaTool.Index] = deltaTool.ID
					}
					if deltaTool.Function.Name != "" {
						toolCallNames[deltaTool.Index] = deltaTool.Function.Name
					}
					if !send(aisdk.StreamEvent{
						Type:  aisdk.StreamEventTypeToolCallDelta,
						Delta: deltaTool.Function.Arguments,
						ToolCall: &aisdk.ToolCall{
							ID:   toolCallIDs[deltaTool.Index],
							Name: toolCallNames[deltaTool.Index],
						},
					}) {
						return
					}
				}
			}

			if len(chunk.Choices) > 0 && chunk.Choices[0].FinishReason != "" {
				finishReason = toFinishReason(chunk.Choices[0].FinishReason)
			}
			if chunk.Usage.TotalTokens > 0 {
				usage = aisdk.Usage{
					InputTokens:  int(chunk.Usage.PromptTokens),
					OutputTokens: int(chunk.Usage.CompletionTokens),
				}
			}
		}

		if err := sdkStream.Err(); err != nil {
			send(aisdk.StreamEvent{Type: aisdk.StreamEventTypeError, Err: mapError(err)})
			return
		}

		send(aisdk.StreamEvent{Type: aisdk.StreamEventTypeFinish, FinishReason: finishReason, Usage: usage})
	}()

	return events, nil
}
