// otel/otel_test.go
package otel_test

import (
	"context"
	"errors"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/habibiramadhan-dev/aisdk-go"
	aisdkotel "github.com/habibiramadhan-dev/aisdk-go/otel"
)

// fakeModel is a minimal aisdk.Model + aisdk.ModelInfo double, controlled
// entirely by the test — no real provider SDK involved.
type fakeModel struct {
	provider, modelName string
	resp                aisdk.GenerateResponse
	err                 error
}

func (m *fakeModel) Provider() string  { return m.provider }
func (m *fakeModel) ModelName() string { return m.modelName }

func (m *fakeModel) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	return m.resp, m.err
}

func (m *fakeModel) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	panic("not used in Generate-path tests")
}

func newTestTracerProvider() (*sdktrace.TracerProvider, *tracetest.InMemoryExporter) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	return tp, exporter
}

func TestWrap_Generate_RecordsGenAIAttributes(t *testing.T) {
	tp, exporter := newTestTracerProvider()
	defer func() { _ = tp.Shutdown(context.Background()) }()

	inner := &fakeModel{
		provider:  "anthropic",
		modelName: "claude-sonnet-5",
		resp: aisdk.GenerateResponse{
			FinishReason: aisdk.FinishReasonStop,
			Usage:        aisdk.Usage{InputTokens: 10, OutputTokens: 5},
		},
	}
	model := aisdkotel.Wrap(inner, aisdkotel.WithTracerProvider(tp))

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{MaxTokens: 64})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	span := spans[0]
	if span.Name != "generate claude-sonnet-5" {
		t.Errorf("span.Name = %q, want %q", span.Name, "generate claude-sonnet-5")
	}

	attrs := map[string]bool{}
	for _, a := range span.Attributes {
		attrs[string(a.Key)] = true
	}
	for _, want := range []string{"gen_ai.system", "gen_ai.request.model", "gen_ai.request.max_tokens", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.response.finish_reason"} {
		if !attrs[want] {
			t.Errorf("span is missing attribute %q; got %v", want, attrs)
		}
	}
	if span.Status.Code != codes.Ok {
		t.Errorf("span.Status.Code = %v, want Ok", span.Status.Code)
	}
}

func TestWrap_Generate_RecordsErrorOnFailure(t *testing.T) {
	tp, exporter := newTestTracerProvider()
	defer func() { _ = tp.Shutdown(context.Background()) }()

	wantErr := &aisdk.Error{Provider: "anthropic", Code: aisdk.ErrorCodeAuthFailed, Retryable: false}
	inner := &fakeModel{provider: "anthropic", modelName: "claude-sonnet-5", err: wantErr}
	model := aisdkotel.Wrap(inner, aisdkotel.WithTracerProvider(tp))

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("Generate error = %v, want it to be %v", err, wantErr)
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("span.Status.Code = %v, want Error", spans[0].Status.Code)
	}
	if len(spans[0].Events) == 0 {
		t.Error("span has no events, want at least one from RecordError")
	}
}

func TestWrap_Generate_NoModelInfo_FallsBackToGenericSpanName(t *testing.T) {
	tp, exporter := newTestTracerProvider()
	defer func() { _ = tp.Shutdown(context.Background()) }()

	// modelWithoutInfo does NOT implement aisdk.ModelInfo.
	inner := &modelWithoutInfo{resp: aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop}}
	model := aisdkotel.Wrap(inner, aisdkotel.WithTracerProvider(tp))

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Name != "aisdk.generate" {
		t.Errorf("span.Name = %q, want %q", spans[0].Name, "aisdk.generate")
	}
	for _, a := range spans[0].Attributes {
		if a.Key == "gen_ai.system" || a.Key == "gen_ai.request.model" {
			t.Errorf("span has %q attribute, want it omitted when ModelInfo isn't implemented", a.Key)
		}
	}
}

type modelWithoutInfo struct {
	resp aisdk.GenerateResponse
}

func (m *modelWithoutInfo) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	return m.resp, nil
}
func (m *modelWithoutInfo) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	panic("not used")
}

func TestWrap_Generate_CapturesContentOnlyWhenOptedIn(t *testing.T) {
	inner := &fakeModel{
		provider:  "anthropic",
		modelName: "claude-sonnet-5",
		resp: aisdk.GenerateResponse{
			Message:      aisdk.Message{Role: aisdk.RoleAssistant, Parts: []aisdk.ContentPart{aisdk.TextPart("hi there")}},
			FinishReason: aisdk.FinishReasonStop,
		},
	}
	req := aisdk.GenerateRequest{Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("secret prompt")}}}}

	t.Run("without WithCaptureContent", func(t *testing.T) {
		tp, exporter := newTestTracerProvider()
		defer func() { _ = tp.Shutdown(context.Background()) }()
		model := aisdkotel.Wrap(inner, aisdkotel.WithTracerProvider(tp))
		if _, err := model.Generate(context.Background(), req); err != nil {
			t.Fatalf("Generate returned error: %v", err)
		}
		for _, a := range exporter.GetSpans()[0].Attributes {
			if a.Key == "gen_ai.prompt" || a.Key == "gen_ai.completion" {
				t.Errorf("span has %q attribute without WithCaptureContent, want it absent", a.Key)
			}
		}
	})

	t.Run("with WithCaptureContent", func(t *testing.T) {
		tp, exporter := newTestTracerProvider()
		defer func() { _ = tp.Shutdown(context.Background()) }()
		model := aisdkotel.Wrap(inner, aisdkotel.WithTracerProvider(tp), aisdkotel.WithCaptureContent())
		if _, err := model.Generate(context.Background(), req); err != nil {
			t.Fatalf("Generate returned error: %v", err)
		}
		attrs := map[string]bool{}
		for _, a := range exporter.GetSpans()[0].Attributes {
			attrs[string(a.Key)] = true
		}
		if !attrs["gen_ai.prompt"] || !attrs["gen_ai.completion"] {
			t.Error("span missing gen_ai.prompt/gen_ai.completion with WithCaptureContent set")
		}
	})
}

func TestWrap_Stream_ClosesSpanOnFinishWithUsage(t *testing.T) {
	tp, exporter := newTestTracerProvider()
	defer func() { _ = tp.Shutdown(context.Background()) }()

	inner := &streamingFakeModel{
		provider:  "openai",
		modelName: "gpt-4o",
		events: []aisdk.StreamEvent{
			{Type: aisdk.StreamEventTypeTextDelta, Delta: "hel"},
			{Type: aisdk.StreamEventTypeTextDelta, Delta: "lo"},
			{Type: aisdk.StreamEventTypeFinish, FinishReason: aisdk.FinishReasonStop, Usage: aisdk.Usage{InputTokens: 3, OutputTokens: 2}},
		},
	}
	model := aisdkotel.Wrap(inner, aisdkotel.WithTracerProvider(tp))

	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	for range stream {
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	span := spans[0]
	if span.Name != "stream gpt-4o" {
		t.Errorf("span.Name = %q, want %q", span.Name, "stream gpt-4o")
	}
	attrs := map[string]attribute.Value{}
	for _, a := range span.Attributes {
		attrs[string(a.Key)] = a.Value
	}
	if v, ok := attrs["gen_ai.usage.input_tokens"]; !ok || v.AsInt64() != 3 {
		t.Errorf("gen_ai.usage.input_tokens = %v, want 3", v)
	}
	if v, ok := attrs["gen_ai.usage.output_tokens"]; !ok || v.AsInt64() != 2 {
		t.Errorf("gen_ai.usage.output_tokens = %v, want 2", v)
	}
	if codes.Ok != span.Status.Code {
		t.Errorf("span.Status.Code = %v, want Ok", span.Status.Code)
	}
}

func TestWrap_Stream_RecordsErrorEvent(t *testing.T) {
	tp, exporter := newTestTracerProvider()
	defer func() { _ = tp.Shutdown(context.Background()) }()

	streamErr := &aisdk.Error{Provider: "openai", Code: aisdk.ErrorCodeServerError, Retryable: true}
	inner := &streamingFakeModel{
		provider:  "openai",
		modelName: "gpt-4o",
		events: []aisdk.StreamEvent{
			{Type: aisdk.StreamEventTypeError, Err: streamErr},
		},
	}
	model := aisdkotel.Wrap(inner, aisdkotel.WithTracerProvider(tp))

	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	for range stream {
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("span.Status.Code = %v, want Error", spans[0].Status.Code)
	}
}

func TestWrap_Stream_ConnectionErrorRecordsAndEndsSpanImmediately(t *testing.T) {
	tp, exporter := newTestTracerProvider()
	defer func() { _ = tp.Shutdown(context.Background()) }()

	connErr := &aisdk.Error{Provider: "openai", Code: aisdk.ErrorCodeAuthFailed, Retryable: false}
	inner := &streamingFakeModel{provider: "openai", modelName: "gpt-4o", connectErr: connErr}
	model := aisdkotel.Wrap(inner, aisdkotel.WithTracerProvider(tp))

	_, err := model.Stream(context.Background(), aisdk.GenerateRequest{})
	if !errors.Is(err, connErr) {
		t.Fatalf("Stream error = %v, want it to be %v", err, connErr)
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1 (the span must still be exported/ended even though the connection itself failed)", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("span.Status.Code = %v, want Error", spans[0].Status.Code)
	}
}

func TestWrap_Stream_CapturesAccumulatedTextOnlyWhenOptedIn(t *testing.T) {
	inner := &streamingFakeModel{
		provider:  "openai",
		modelName: "gpt-4o",
		events: []aisdk.StreamEvent{
			{Type: aisdk.StreamEventTypeTextDelta, Delta: "hel"},
			{Type: aisdk.StreamEventTypeTextDelta, Delta: "lo"},
			{Type: aisdk.StreamEventTypeFinish, FinishReason: aisdk.FinishReasonStop},
		},
	}

	tp, exporter := newTestTracerProvider()
	defer func() { _ = tp.Shutdown(context.Background()) }()
	model := aisdkotel.Wrap(inner, aisdkotel.WithTracerProvider(tp), aisdkotel.WithCaptureContent())

	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	for range stream {
	}

	for _, a := range exporter.GetSpans()[0].Attributes {
		if a.Key == "gen_ai.completion" {
			if a.Value.AsString() != "hello" {
				t.Errorf("gen_ai.completion = %q, want %q", a.Value.AsString(), "hello")
			}
			return
		}
	}
	t.Error("span missing gen_ai.completion with WithCaptureContent set")
}

// streamingFakeModel is a minimal aisdk.Model + aisdk.ModelInfo double for
// the Stream path — connectErr simulates a failure at the initial
// model.Stream(...) call itself; events is what a successful connection
// yields through the channel.
type streamingFakeModel struct {
	provider, modelName string
	connectErr          error
	events              []aisdk.StreamEvent
}

func (m *streamingFakeModel) Provider() string  { return m.provider }
func (m *streamingFakeModel) ModelName() string { return m.modelName }

func (m *streamingFakeModel) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	panic("not used in Stream-path tests")
}

func (m *streamingFakeModel) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	if m.connectErr != nil {
		return nil, m.connectErr
	}
	events := make(chan aisdk.StreamEvent, len(m.events))
	for _, e := range m.events {
		events <- e
	}
	close(events)
	return events, nil
}

func TestWrap_ComposesWithFallback_RecordsNestedFallbackEvents(t *testing.T) {
	tp, exporter := newTestTracerProvider()
	defer func() { _ = tp.Shutdown(context.Background()) }()

	m1 := &fakeModel{
		provider:  "anthropic",
		modelName: "claude-sonnet-5",
		err:       &aisdk.Error{Provider: "anthropic", Code: aisdk.ErrorCodeOverloaded, Retryable: true},
	}
	m2 := &fakeModel{
		provider:  "openai",
		modelName: "gpt-4o",
		resp:      aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop, Usage: aisdk.Usage{InputTokens: 1, OutputTokens: 1}},
	}

	fallback := aisdk.Fallback([]aisdk.Model{m1, m2})
	model := aisdkotel.Wrap(fallback, aisdkotel.WithTracerProvider(tp))

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want exactly 1 — otel.Wrap starts one span per call, regardless of how many models Fallback tries underneath", len(spans))
	}
	span := spans[0]

	// otel.Wrap couldn't have populated gen_ai.system/gen_ai.request.model —
	// Fallback doesn't implement aisdk.ModelInfo (it wraps MULTIPLE models,
	// so it has no single identity), confirming the graceful-degradation
	// path from Task 3 is exercised here, not just tested in isolation.
	for _, a := range span.Attributes {
		if a.Key == "gen_ai.system" || a.Key == "gen_ai.request.model" {
			t.Errorf("span has %q attribute; Fallback has no single ModelInfo identity, want this omitted", a.Key)
		}
	}
	if span.Name != "aisdk.generate" {
		t.Errorf("span.Name = %q, want %q (the no-ModelInfo fallback name)", span.Name, "aisdk.generate")
	}

	if len(span.Events) == 0 {
		t.Fatal("span has no events — Fallback's fallback.attempt event should have landed on this same span via the ambient-context pattern")
	}
	foundOverloadedSkip := false
	for _, e := range span.Events {
		if e.Name != "fallback.attempt" {
			continue
		}
		for _, a := range e.Attributes {
			if a.Key == "aisdk.fallback.outcome" && a.Value.AsString() == "skip_overloaded" {
				foundOverloadedSkip = true
			}
		}
	}
	if !foundOverloadedSkip {
		t.Error("no fallback.attempt event with outcome=skip_overloaded found — m1's Overloaded error should have produced one")
	}
}
