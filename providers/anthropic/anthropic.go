package anthropic

import (
	"context"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/habibiramadhan-dev/aisdk-go"
)

// Provider constructs Anthropic-backed aisdk.Model values.
type Provider struct {
	client anthropicsdk.Client
}

// New constructs a Provider. Extra option.RequestOption values (e.g.
// option.WithBaseURL for tests) are applied after the API key.
func New(apiKey string, opts ...option.RequestOption) *Provider {
	allOpts := append([]option.RequestOption{option.WithAPIKey(apiKey)}, opts...)
	return &Provider{client: anthropicsdk.NewClient(allOpts...)}
}

// Model returns an aisdk.Model bound to the given Anthropic model name
// (e.g. "claude-sonnet-5").
func (p *Provider) Model(name string) aisdk.Model {
	return &model{client: p.client, modelName: name}
}

type model struct {
	client    anthropicsdk.Client
	modelName string
}

// Provider implements aisdk.ModelInfo.
func (m *model) Provider() string { return "anthropic" }

// ModelName implements aisdk.ModelInfo.
func (m *model) ModelName() string { return m.modelName }

func (m *model) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	params := toMessageNewParams(m.modelName, req)

	msg, err := m.client.Messages.New(ctx, params)
	if err != nil {
		return aisdk.GenerateResponse{}, mapError(err)
	}

	return toGenerateResponse(msg), nil
}

// Stream's returned channel is closed once the response ends, errors, or ctx
// is cancelled. Callers that stop ranging over the channel before it closes
// on its own MUST cancel ctx — otherwise the background goroutine blocks
// forever trying to send the next event to a channel nobody is reading.
func (m *model) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	params := toMessageNewParams(m.modelName, req)
	sdkStream := m.client.Messages.NewStreaming(ctx, params)

	events := make(chan aisdk.StreamEvent)
	go func() {
		defer close(events)
		defer func() { _ = sdkStream.Close() }()

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
			event := sdkStream.Current()

			if se, ok := toStreamEvent(event); ok {
				if !send(se) {
					return
				}
				continue
			}

			switch event.Type {
			case "content_block_start":
				if event.ContentBlock.Type != "tool_use" {
					continue
				}
				toolCallIDs[event.Index] = event.ContentBlock.ID
				toolCallNames[event.Index] = event.ContentBlock.Name
				if !send(aisdk.StreamEvent{
					Type:     aisdk.StreamEventTypeToolCallDelta,
					ToolCall: &aisdk.ToolCall{ID: event.ContentBlock.ID, Name: event.ContentBlock.Name},
				}) {
					return
				}
			case "content_block_delta":
				if event.Delta.Type != "input_json_delta" {
					continue
				}
				if !send(aisdk.StreamEvent{
					Type:  aisdk.StreamEventTypeToolCallDelta,
					Delta: event.Delta.PartialJSON,
					ToolCall: &aisdk.ToolCall{
						ID:   toolCallIDs[event.Index],
						Name: toolCallNames[event.Index],
					},
				}) {
					return
				}
			case "message_delta":
				finishReason = toFinishReason(event.Delta.StopReason)
				usage = aisdk.Usage{
					InputTokens:  int(event.Usage.InputTokens),
					OutputTokens: int(event.Usage.OutputTokens),
				}
			case "message_stop":
				if !send(aisdk.StreamEvent{Type: aisdk.StreamEventTypeFinish, FinishReason: finishReason, Usage: usage}) {
					return
				}
			}
		}

		if err := sdkStream.Err(); err != nil {
			send(aisdk.StreamEvent{Type: aisdk.StreamEventTypeError, Err: mapError(err)})
		}
	}()

	return events, nil
}

