package aisdk

import (
	"context"
	"errors"
	"testing"
)

// fakeModel is a test double satisfying the Model interface without any network.
type fakeModel struct {
	lastRequest GenerateRequest
	response    GenerateResponse
	err         error
}

func (f *fakeModel) Generate(ctx context.Context, req GenerateRequest) (GenerateResponse, error) {
	f.lastRequest = req
	return f.response, f.err
}

func (f *fakeModel) Stream(ctx context.Context, req GenerateRequest) (<-chan StreamEvent, error) {
	return nil, errors.New("not used in this test")
}

func TestAsk_SendsSingleUserMessageAndReturnsText(t *testing.T) {
	model := &fakeModel{
		response: GenerateResponse{
			Message: Message{
				Role:  RoleAssistant,
				Parts: []ContentPart{TextPart("hi there")},
			},
			FinishReason: FinishReasonStop,
		},
	}

	got, err := Ask(context.Background(), model, "hello")
	if err != nil {
		t.Fatalf("Ask returned error: %v", err)
	}
	if got != "hi there" {
		t.Errorf("Ask() = %q, want %q", got, "hi there")
	}

	if len(model.lastRequest.Messages) != 1 {
		t.Fatalf("Generate called with %d messages, want 1", len(model.lastRequest.Messages))
	}
	sent := model.lastRequest.Messages[0]
	if sent.Role != RoleUser {
		t.Errorf("sent message role = %q, want %q", sent.Role, RoleUser)
	}
	if len(sent.Parts) != 1 || sent.Parts[0].Text != "hello" {
		t.Errorf("sent message parts = %+v, want a single text part %q", sent.Parts, "hello")
	}
	if model.lastRequest.MaxTokens <= 0 {
		t.Errorf("sent MaxTokens = %d, want a positive default (providers reject max_tokens <= 0)", model.lastRequest.MaxTokens)
	}
}

func TestAsk_PropagatesModelError(t *testing.T) {
	wantErr := errors.New("provider unavailable")
	model := &fakeModel{err: wantErr}

	_, err := Ask(context.Background(), model, "hello")
	if !errors.Is(err, wantErr) {
		t.Errorf("Ask() error = %v, want it to wrap %v", err, wantErr)
	}
}
