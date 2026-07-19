package anthropic

import (
	"context"
	"errors"

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

func (m *model) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	params := toMessageNewParams(m.modelName, req)

	msg, err := m.client.Messages.New(ctx, params)
	if err != nil {
		return aisdk.GenerateResponse{}, mapError(err)
	}

	return toGenerateResponse(msg), nil
}

// ErrStreamingNotImplemented is returned by Stream until Fase 2 implements it.
var ErrStreamingNotImplemented = errors.New("aisdk/anthropic: streaming not implemented until Fase 2")

func (m *model) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	return nil, ErrStreamingNotImplemented
}
