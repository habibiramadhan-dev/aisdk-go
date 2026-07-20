// fallback_test.go
package aisdk_test

import (
	"context"
	"errors"
	"testing"
	"time"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/habibiramadhan-dev/aisdk-go"
)

// countingModel records every Generate/Stream call and returns canned
// responses in order — errs[i]/streamErrs[i] for the (i+1)th call, then the
// success value once the corresponding errs slice is exhausted. A nil entry
// means "succeed on this call."
type countingModel struct {
	name         string
	errs         []error
	successResp  aisdk.GenerateResponse
	streamErrs   []error
	streamEvents []aisdk.StreamEvent
	calls        int
	streamCalls  int
}

func (m *countingModel) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	i := m.calls
	m.calls++
	if i < len(m.errs) {
		return aisdk.GenerateResponse{}, m.errs[i]
	}
	return m.successResp, nil
}

func (m *countingModel) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	i := m.streamCalls
	m.streamCalls++
	if i < len(m.streamErrs) {
		return nil, m.streamErrs[i]
	}
	events := make(chan aisdk.StreamEvent, len(m.streamEvents))
	for _, e := range m.streamEvents {
		events <- e
	}
	close(events)
	return events, nil
}

func retryableErr(code aisdk.ErrorCode) error {
	return &aisdk.Error{Provider: "fake", Code: code, Retryable: true}
}

func nonRetryableErr(code aisdk.ErrorCode) error {
	return &aisdk.Error{Provider: "fake", Code: code, Retryable: false}
}

var noWaitBackoff = func(attempt int) time.Duration { return 0 }

func TestFallback_FirstModelSucceedsImmediately(t *testing.T) {
	m1 := &countingModel{name: "m1", successResp: aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop}}
	m2 := &countingModel{name: "m2", successResp: aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop}}

	model := aisdk.Fallback([]aisdk.Model{m1, m2})
	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if m1.calls != 1 {
		t.Errorf("m1.calls = %d, want 1", m1.calls)
	}
	if m2.calls != 0 {
		t.Errorf("m2.calls = %d, want 0 (should never be tried)", m2.calls)
	}
}

func TestFallback_RetriesRetryableErrorThenSucceeds(t *testing.T) {
	m1 := &countingModel{
		name:        "m1",
		errs:        []error{retryableErr(aisdk.ErrorCodeRateLimited), retryableErr(aisdk.ErrorCodeRateLimited)},
		successResp: aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop},
	}
	m2 := &countingModel{name: "m2"}

	model := aisdk.Fallback([]aisdk.Model{m1, m2}, aisdk.WithMaxRetries(3), aisdk.WithBackoff(noWaitBackoff))
	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if m1.calls != 3 {
		t.Errorf("m1.calls = %d, want 3 (2 failures + 1 success)", m1.calls)
	}
	if m2.calls != 0 {
		t.Errorf("m2.calls = %d, want 0", m2.calls)
	}
}

func TestFallback_FallsBackAfterExhaustingRetries(t *testing.T) {
	m1 := &countingModel{
		name: "m1",
		errs: []error{
			retryableErr(aisdk.ErrorCodeRateLimited),
			retryableErr(aisdk.ErrorCodeRateLimited),
			retryableErr(aisdk.ErrorCodeRateLimited),
		},
	}
	m2 := &countingModel{name: "m2", successResp: aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop}}

	model := aisdk.Fallback([]aisdk.Model{m1, m2}, aisdk.WithMaxRetries(2), aisdk.WithBackoff(noWaitBackoff))
	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if m1.calls != 3 {
		t.Errorf("m1.calls = %d, want 3 (1 initial + 2 retries, all failing)", m1.calls)
	}
	if m2.calls != 1 {
		t.Errorf("m2.calls = %d, want 1", m2.calls)
	}
}

func TestFallback_OverloadedSkipsRetryFallsBackImmediately(t *testing.T) {
	m1 := &countingModel{name: "m1", errs: []error{retryableErr(aisdk.ErrorCodeOverloaded)}}
	m2 := &countingModel{name: "m2", successResp: aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop}}

	model := aisdk.Fallback([]aisdk.Model{m1, m2}, aisdk.WithMaxRetries(5), aisdk.WithBackoff(noWaitBackoff))
	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if m1.calls != 1 {
		t.Errorf("m1.calls = %d, want 1 (Overloaded must not be retried on the same model)", m1.calls)
	}
	if m2.calls != 1 {
		t.Errorf("m2.calls = %d, want 1", m2.calls)
	}
}

func TestFallback_NonRetryableFailsImmediatelyNoFallback(t *testing.T) {
	m1 := &countingModel{name: "m1", errs: []error{nonRetryableErr(aisdk.ErrorCodeAuthFailed)}}
	m2 := &countingModel{name: "m2"}

	model := aisdk.Fallback([]aisdk.Model{m1, m2})
	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})

	var aisdkErr *aisdk.Error
	if !errors.As(err, &aisdkErr) {
		t.Fatalf("Generate error = %v, want it to unwrap to *aisdk.Error", err)
	}
	if aisdkErr.Code != aisdk.ErrorCodeAuthFailed {
		t.Errorf("aisdkErr.Code = %q, want %q", aisdkErr.Code, aisdk.ErrorCodeAuthFailed)
	}
	if m2.calls != 0 {
		t.Errorf("m2.calls = %d, want 0 (non-retryable errors must not fall back)", m2.calls)
	}
}

func TestFallback_AllModelsExhausted_ReturnsJoinedError(t *testing.T) {
	m1 := &countingModel{name: "m1", errs: []error{retryableErr(aisdk.ErrorCodeRateLimited)}}
	m2 := &countingModel{name: "m2", errs: []error{retryableErr(aisdk.ErrorCodeRateLimited)}}

	model := aisdk.Fallback([]aisdk.Model{m1, m2}, aisdk.WithMaxRetries(0), aisdk.WithBackoff(noWaitBackoff))
	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	if err == nil {
		t.Fatal("Generate returned nil error, want a joined error from both models")
	}
	if m1.calls != 1 || m2.calls != 1 {
		t.Errorf("m1.calls=%d m2.calls=%d, want 1 and 1", m1.calls, m2.calls)
	}
	// errors.As must still work through the joined error.
	var aisdkErr *aisdk.Error
	if !errors.As(err, &aisdkErr) {
		t.Fatal("errors.As failed to find *aisdk.Error inside the joined error")
	}
}

func TestFallback_HonorsRetryAfterOverBackoff(t *testing.T) {
	retryAfterErr := &aisdk.Error{Provider: "fake", Code: aisdk.ErrorCodeRateLimited, Retryable: true, RetryAfter: 30 * time.Millisecond}
	m1 := &countingModel{
		name:        "m1",
		errs:        []error{retryAfterErr},
		successResp: aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop},
	}

	slowBackoff := func(attempt int) time.Duration { return 5 * time.Second }
	model := aisdk.Fallback([]aisdk.Model{m1}, aisdk.WithMaxRetries(1), aisdk.WithBackoff(slowBackoff))

	start := time.Now()
	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if elapsed > 500*time.Millisecond {
		t.Errorf("Generate took %v, want it to honor RetryAfter (30ms) rather than the 5s backoff", elapsed)
	}
}

func TestFallback_RespectsBudget(t *testing.T) {
	m1 := &countingModel{
		name: "m1",
		errs: []error{
			retryableErr(aisdk.ErrorCodeRateLimited),
			retryableErr(aisdk.ErrorCodeRateLimited),
			retryableErr(aisdk.ErrorCodeRateLimited),
		},
	}
	m2 := &countingModel{name: "m2"}

	longBackoff := func(attempt int) time.Duration { return 200 * time.Millisecond }
	model := aisdk.Fallback([]aisdk.Model{m1, m2}, aisdk.WithMaxRetries(10), aisdk.WithBackoff(longBackoff), aisdk.WithBudget(50*time.Millisecond))

	start := time.Now()
	_, _ = model.Generate(context.Background(), aisdk.GenerateRequest{})
	elapsed := time.Since(start)
	if elapsed > 300*time.Millisecond {
		t.Errorf("Generate took %v, want the 50ms budget to cut retries short well before 10 retries at 200ms each (2s+)", elapsed)
	}
}

func TestFallback_PanicsOnEmptyModelList(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Fallback did not panic on an empty models slice")
		}
	}()
	aisdk.Fallback(nil)
}

func TestFallback_Stream_RetriesConnectionThenSucceeds(t *testing.T) {
	m1 := &countingModel{
		name:         "m1",
		streamErrs:   []error{retryableErr(aisdk.ErrorCodeRateLimited)},
		streamEvents: []aisdk.StreamEvent{{Type: aisdk.StreamEventTypeFinish, FinishReason: aisdk.FinishReasonStop}},
	}

	model := aisdk.Fallback([]aisdk.Model{m1}, aisdk.WithMaxRetries(1), aisdk.WithBackoff(noWaitBackoff))
	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	var events []aisdk.StreamEvent
	for e := range stream {
		events = append(events, e)
	}
	if len(events) != 1 || events[0].Type != aisdk.StreamEventTypeFinish {
		t.Errorf("events = %+v, want a single Finish event", events)
	}
	if m1.streamCalls != 2 {
		t.Errorf("m1.streamCalls = %d, want 2 (1 failed connection + 1 successful)", m1.streamCalls)
	}
}

func TestFallback_Stream_FallsBackToNextModel(t *testing.T) {
	// Overloaded, not a non-retryable error: per the classification rules
	// (see Design decisions), a non-retryable error is FATAL — it stops the
	// whole chain immediately rather than falling back (already covered by
	// TestFallback_NonRetryableFailsImmediatelyNoFallback in Task 4).
	// Overloaded is the case that actually skips to the next model, so it's
	// the correct fixture for a test named "falls back to the next model."
	m1 := &countingModel{name: "m1", streamErrs: []error{retryableErr(aisdk.ErrorCodeOverloaded)}}
	m2 := &countingModel{
		name:         "m2",
		streamEvents: []aisdk.StreamEvent{{Type: aisdk.StreamEventTypeFinish, FinishReason: aisdk.FinishReasonStop}},
	}

	model := aisdk.Fallback([]aisdk.Model{m1, m2})
	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	for range stream {
	}
	if m1.streamCalls != 1 || m2.streamCalls != 1 {
		t.Errorf("m1.streamCalls=%d m2.streamCalls=%d, want 1 and 1", m1.streamCalls, m2.streamCalls)
	}
}

func TestFallback_Stream_DoesNotInterceptMidStreamErrors(t *testing.T) {
	m1 := &countingModel{
		name: "m1",
		streamEvents: []aisdk.StreamEvent{
			{Type: aisdk.StreamEventTypeTextDelta, Delta: "partial"},
			{Type: aisdk.StreamEventTypeError, Err: retryableErr(aisdk.ErrorCodeServerError)},
		},
	}

	model := aisdk.Fallback([]aisdk.Model{m1})
	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	var sawError bool
	for e := range stream {
		if e.Type == aisdk.StreamEventTypeError {
			sawError = true
		}
	}
	if !sawError {
		t.Fatal("never saw the mid-stream Error event — test setup is broken")
	}
	// The critical assertion: Fallback must NOT have retried the connection
	// in response to the mid-stream error — it only governs the initial
	// Stream() call, per design.md §6.
	if m1.streamCalls != 1 {
		t.Errorf("m1.streamCalls = %d, want 1 (Fallback must not react to a mid-stream StreamEvent{Type: Error})", m1.streamCalls)
	}
}

func TestFallback_RecordsSpanEventForRetry(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	defer func() { _ = tp.Shutdown(context.Background()) }()
	tracer := tp.Tracer("test")

	m1 := &countingModel{
		name:        "m1",
		errs:        []error{retryableErr(aisdk.ErrorCodeRateLimited)},
		successResp: aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop},
	}

	ctx, span := tracer.Start(context.Background(), "test-span")
	model := aisdk.Fallback([]aisdk.Model{m1}, aisdk.WithMaxRetries(1), aisdk.WithBackoff(noWaitBackoff))
	_, err := model.Generate(ctx, aisdk.GenerateRequest{})
	span.End()
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	events := spans[0].Events
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0].Name != "fallback.attempt" {
		t.Errorf("events[0].Name = %q, want %q", events[0].Name, "fallback.attempt")
	}
	foundRetryCount, foundOutcome := false, false
	for _, attr := range events[0].Attributes {
		if attr.Key == "retry.count" {
			foundRetryCount = true
			if attr.Value.AsInt64() != 0 {
				t.Errorf("retry.count = %v, want 0", attr.Value.AsInt64())
			}
		}
		if attr.Key == "aisdk.fallback.outcome" {
			foundOutcome = true
			if attr.Value.AsString() != "retry" {
				t.Errorf("aisdk.fallback.outcome = %q, want %q", attr.Value.AsString(), "retry")
			}
		}
	}
	if !foundRetryCount {
		t.Error("event has no retry.count attribute")
	}
	if !foundOutcome {
		t.Error("event has no aisdk.fallback.outcome attribute")
	}
}

func TestFallback_NoSpanInContext_DoesNotPanic(t *testing.T) {
	// Ambient-context contract: Fallback must work identically whether or
	// not a span is present in ctx — trace.SpanFromContext on a plain
	// context.Background() returns a real, harmless no-op Span.
	m1 := &countingModel{
		name:        "m1",
		errs:        []error{retryableErr(aisdk.ErrorCodeRateLimited)},
		successResp: aisdk.GenerateResponse{FinishReason: aisdk.FinishReasonStop},
	}
	model := aisdk.Fallback([]aisdk.Model{m1}, aisdk.WithMaxRetries(1), aisdk.WithBackoff(noWaitBackoff))
	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
}
