// providers/gemini/gemini.go
package gemini

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/habibiramadhan-dev/aisdk-go"
	genaisdk "google.golang.org/genai"
)

// Provider constructs Gemini-backed aisdk.Model values.
type Provider struct {
	client *genaisdk.Client
}

// New constructs a Provider against the Gemini Developer API backend. Unlike
// the Anthropic/OpenAI adapters' New, this one takes a context and can fail —
// genai.NewClient itself is context-aware and fallible, so this adapter's New
// honestly reflects that rather than papering over it.
//
// configure functions run after APIKey is set on the ClientConfig, letting
// callers override fields like HTTPOptions.BaseURL (used by tests to point
// at an httptest.Server).
func New(ctx context.Context, apiKey string, configure ...func(*genaisdk.ClientConfig)) (*Provider, error) {
	cc := &genaisdk.ClientConfig{APIKey: apiKey}
	for _, fn := range configure {
		fn(cc)
	}

	client, err := genaisdk.NewClient(ctx, cc)
	if err != nil {
		return nil, err
	}
	return &Provider{client: client}, nil
}

// Model returns an aisdk.Model bound to the given Gemini model name
// (e.g. "gemini-2.0-flash").
func (p *Provider) Model(name string) aisdk.Model {
	return &model{client: p.client, modelName: name}
}

type model struct {
	client    *genaisdk.Client
	modelName string
}

// Provider implements aisdk.ModelInfo.
func (m *model) Provider() string { return "gemini" }

// ModelName implements aisdk.ModelInfo.
func (m *model) ModelName() string { return m.modelName }

func (m *model) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	contents, config := toGenerateContentParams(req)

	resp, err := m.client.Models.GenerateContent(ctx, m.modelName, contents, config)
	if err != nil {
		return aisdk.GenerateResponse{}, mapError(err)
	}

	return toGenerateResponse(resp)
}

// Stream's returned channel is closed once the response ends, errors, or ctx
// is cancelled. Callers that stop ranging over the channel before it closes
// on its own MUST cancel ctx — otherwise the background goroutine blocks
// forever trying to send the next event to a channel nobody is reading.
//
// Gemini's underlying iterator has an asymmetry the other two adapters don't:
// once streaming has started, a canceled ctx does not surface as an error
// from the iterator — the range loop just silently stops yielding (confirmed
// against the real SDK source and a live test during planning). So after the
// loop ends, this method explicitly checks ctx.Err(): a non-nil value means
// the stream was cut off by cancellation (channel just closes, no Finish, no
// Error — matching the cancellation contract above), while nil means the
// stream ended normally (Finish is sent).
func (m *model) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	contents, config := toGenerateContentParams(req)

	events := make(chan aisdk.StreamEvent)
	go func() {
		defer close(events)

		var finishReason aisdk.FinishReason
		var usage aisdk.Usage
		var sawToolCall bool

		send := func(e aisdk.StreamEvent) bool {
			select {
			case events <- e:
				return true
			case <-ctx.Done():
				return false
			}
		}

		for chunk, err := range m.client.Models.GenerateContentStream(ctx, m.modelName, contents, config) {
			if err != nil {
				send(aisdk.StreamEvent{Type: aisdk.StreamEventTypeError, Err: mapError(err)})
				return
			}

			if se, ok := toStreamEvent(chunk); ok {
				if !send(se) {
					return
				}
			}

			if len(chunk.Candidates) > 0 && chunk.Candidates[0].Content != nil {
				for i, part := range chunk.Candidates[0].Content.Parts {
					if part.FunctionCall == nil {
						continue
					}
					sawToolCall = true
					argsJSON, err := json.Marshal(part.FunctionCall.Args)
					if err != nil {
						send(aisdk.StreamEvent{Type: aisdk.StreamEventTypeError, Err: fmt.Errorf("gemini: marshaling function call args: %w", err)})
						return
					}
					if !send(aisdk.StreamEvent{
						Type:  aisdk.StreamEventTypeToolCallDelta,
						Delta: string(argsJSON),
						ToolCall: &aisdk.ToolCall{
							ID:   fmt.Sprintf("gemini-tool-call-%d", i),
							Name: part.FunctionCall.Name,
						},
					}) {
						return
					}
				}
			}

			if len(chunk.Candidates) > 0 && chunk.Candidates[0].FinishReason != "" {
				finishReason = toFinishReason(chunk.Candidates[0].FinishReason)
			}
			if chunk.UsageMetadata != nil {
				usage = aisdk.Usage{
					InputTokens:  int(chunk.UsageMetadata.PromptTokenCount),
					OutputTokens: int(chunk.UsageMetadata.CandidatesTokenCount),
				}
			}
		}

		if ctx.Err() != nil {
			return
		}

		if sawToolCall {
			finishReason = aisdk.FinishReasonToolCalls
		}
		send(aisdk.StreamEvent{Type: aisdk.StreamEventTypeFinish, FinishReason: finishReason, Usage: usage})
	}()

	return events, nil
}
