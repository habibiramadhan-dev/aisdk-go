// structured_test.go
package aisdk_test

import (
	"context"
	"errors"
	"testing"

	"github.com/habibiramadhan-dev/aisdk-go"
)

type weatherReport struct {
	City        string  `json:"city"`
	TemperatureC float64 `json:"temperature_c"`
}

// fakeModel is a minimal aisdk.Model whose Generate response and captured
// request are both controlled by the test — no provider SDK involved.
type fakeModel struct {
	responseText   string
	err            error
	capturedReq    aisdk.GenerateRequest
}

func (m *fakeModel) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	m.capturedReq = req
	if m.err != nil {
		return aisdk.GenerateResponse{}, m.err
	}
	return aisdk.GenerateResponse{
		Message: aisdk.Message{
			Role:  aisdk.RoleAssistant,
			Parts: []aisdk.ContentPart{aisdk.TextPart(m.responseText)},
		},
		FinishReason: aisdk.FinishReasonStop,
	}, nil
}

func (m *fakeModel) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	panic("GenerateStructured must not call Stream")
}

func TestGenerateStructured_ReturnsTypedResult(t *testing.T) {
	model := &fakeModel{responseText: `{"city":"Paris","temperature_c":18.5}`}

	result, err := aisdk.GenerateStructured[weatherReport](context.Background(), model, aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("What's the weather in Paris?")}}},
	})
	if err != nil {
		t.Fatalf("GenerateStructured returned error: %v", err)
	}

	if result.City != "Paris" {
		t.Errorf("result.City = %q, want %q", result.City, "Paris")
	}
	if result.TemperatureC != 18.5 {
		t.Errorf("result.TemperatureC = %v, want 18.5", result.TemperatureC)
	}

	if model.capturedReq.ResponseSchema == nil {
		t.Fatal("GenerateStructured did not set req.ResponseSchema before calling Generate")
	}
}

func TestGenerateStructured_PropagatesGenerateError(t *testing.T) {
	wantErr := errors.New("boom")
	model := &fakeModel{err: wantErr}

	_, err := aisdk.GenerateStructured[weatherReport](context.Background(), model, aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("GenerateStructured error = %v, want it to wrap %v", err, wantErr)
	}
}

func TestGenerateStructured_ReturnsErrorOnInvalidJSON(t *testing.T) {
	model := &fakeModel{responseText: "this is not JSON"}

	_, err := aisdk.GenerateStructured[weatherReport](context.Background(), model, aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
	})
	if err == nil {
		t.Fatal("GenerateStructured returned nil error for a non-JSON response")
	}
}
