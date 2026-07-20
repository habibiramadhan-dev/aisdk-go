// fallback.go
package aisdk

import (
	"context"
	"errors"
	"math/rand"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// FallbackOption configures a Fallback-wrapped Model.
type FallbackOption func(*fallbackConfig)

// WithMaxRetries sets how many additional attempts Fallback makes on the
// SAME model after a retryable error, before moving to the next model in
// the chain. 0 means no retries — a single failure moves straight to
// fallback. Default: 2.
func WithMaxRetries(n int) FallbackOption {
	return func(c *fallbackConfig) { c.maxRetries = n }
}

// WithBackoff sets the wait-time function used between retry attempts on
// the same model. It's called with the zero-based attempt number (0 for the
// wait before the first retry). Ignored for a given attempt when the
// failing error carries a non-zero RetryAfter — that value is honored
// instead. Default: full-jitter exponential backoff, 500ms base, 10s cap.
func WithBackoff(fn func(attempt int) time.Duration) FallbackOption {
	return func(c *fallbackConfig) { c.backoff = fn }
}

// WithBudget sets the overall wall-clock time budget for the whole fallback
// chain — retries and fallbacks across every model combined, not a
// per-model limit. Once the budget is exhausted, Fallback stops trying
// further models/retries and returns whatever errors it has collected so
// far. This exists so a stuck request can't retry-storm across every
// configured provider for an unbounded amount of time. Default: 2 minutes.
func WithBudget(d time.Duration) FallbackOption {
	return func(c *fallbackConfig) { c.budget = d }
}

type fallbackConfig struct {
	maxRetries int
	backoff    func(attempt int) time.Duration
	budget     time.Duration
}

func defaultFallbackConfig() fallbackConfig {
	return fallbackConfig{
		maxRetries: 2,
		backoff:    defaultBackoff,
		budget:     2 * time.Minute,
	}
}

// defaultBackoff is full-jitter exponential backoff (base 500ms, doubling
// per attempt, capped at 10s): a well-known algorithm for avoiding
// synchronized/thundering-herd retries across many concurrent callers,
// better than fixed or non-jittered exponential delay.
func defaultBackoff(attempt int) time.Duration {
	const base = 500 * time.Millisecond
	const capDur = 10 * time.Second
	d := base << attempt
	if d <= 0 || d > capDur { // d<=0 guards against overflow at a large attempt count
		d = capDur
	}
	return time.Duration(rand.Int63n(int64(d)))
}

// Fallback wraps models so that a Generate/Stream call tries modelA first,
// retrying transient failures on it with backoff before moving to modelB,
// and so on. See design.md §6 for the full retry/fallback contract this
// implements — in short: a Retryable error (other than Overloaded) is
// retried on the same model up to WithMaxRetries times; ErrorCodeOverloaded
// skips straight to the next model without burning retries on this one; a
// non-Retryable error (or any error that doesn't unwrap to *aisdk.Error at
// all) fails the whole chain immediately, since those tend to fail
// identically on every provider. If every model is exhausted, the returned
// error is errors.Join of every attempt's error — errors.As/errors.Is still
// work through it.
//
// Fallback panics if models is empty — there is no sane way to satisfy a
// Generate/Stream call with zero models, and returning a Model that would
// always silently misbehave (an empty errors.Join is nil, which would look
// like success) is worse than failing loudly at construction time.
//
// Fallback only covers Generate calls and the initial connection for
// Stream — once Stream has returned a channel to the caller, Fallback does
// not intercept, retry, or replay anything flowing through it; a mid-stream
// failure surfaces as StreamEvent{Type: Error} exactly as it would from a
// bare adapter, and retrying the whole call is left to the caller.
func Fallback(models []Model, opts ...FallbackOption) Model {
	if len(models) == 0 {
		panic("aisdk: Fallback called with zero models")
	}
	cfg := defaultFallbackConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return &fallbackModel{models: models, cfg: cfg}
}

type fallbackModel struct {
	models []Model
	cfg    fallbackConfig
}

func (f *fallbackModel) Generate(ctx context.Context, req GenerateRequest) (GenerateResponse, error) {
	deadline := time.Now().Add(f.cfg.budget)

	var errs []error
	for _, model := range f.models {
		resp, err, fatal := f.tryModelGenerate(ctx, deadline, model, req)
		if err == nil {
			return resp, nil
		}
		errs = append(errs, err)
		if fatal || !time.Now().Before(deadline) || ctx.Err() != nil {
			break
		}
	}
	return GenerateResponse{}, errors.Join(errs...)
}

// tryModelGenerate retries a single model per the classification rules
// documented on Fallback, returning fatal=true only when the whole chain
// should stop (non-retryable error, unclassified error, or the caller's own
// ctx was cancelled) — false covers both "succeeded" and "exhausted this
// model, try the next one."
func (f *fallbackModel) tryModelGenerate(ctx context.Context, deadline time.Time, model Model, req GenerateRequest) (GenerateResponse, error, bool) {
	for attempt := 0; ; attempt++ {
		resp, err := model.Generate(ctx, req)
		if err == nil {
			return resp, nil, false
		}

		var aisdkErr *Error
		if !errors.As(err, &aisdkErr) {
			return GenerateResponse{}, err, true
		}
		if !aisdkErr.Retryable {
			return GenerateResponse{}, err, true
		}
		if aisdkErr.Code == ErrorCodeOverloaded {
			recordFallbackAttempt(ctx, attempt, "skip_overloaded")
			return GenerateResponse{}, err, false
		}
		if attempt >= f.cfg.maxRetries {
			recordFallbackAttempt(ctx, attempt, "retries_exhausted")
			return GenerateResponse{}, err, false
		}

		wait := f.cfg.backoff(attempt)
		if aisdkErr.RetryAfter > 0 {
			wait = aisdkErr.RetryAfter
		}
		if wait > time.Until(deadline) {
			recordFallbackAttempt(ctx, attempt, "budget_exhausted")
			return GenerateResponse{}, err, false
		}

		recordFallbackAttempt(ctx, attempt, "retry")
		select {
		case <-ctx.Done():
			return GenerateResponse{}, err, true
		case <-time.After(wait):
		}
	}
}

// recordFallbackAttempt adds a fallback.attempt event to whatever span is
// currently active in ctx — a harmless no-op if there isn't one (e.g.
// Fallback used without otel.Wrap). This is the ambient-context pattern:
// fallback.go has zero import of, or coupling to, the otel package; it only
// depends on the universal OTel trace API, which is designed to be safe to
// call unconditionally.
func recordFallbackAttempt(ctx context.Context, attempt int, outcome string) {
	trace.SpanFromContext(ctx).AddEvent("fallback.attempt", trace.WithAttributes(
		attribute.Int("retry.count", attempt),
		attribute.String("aisdk.fallback.outcome", outcome),
	))
}

func (f *fallbackModel) Stream(ctx context.Context, req GenerateRequest) (<-chan StreamEvent, error) {
	deadline := time.Now().Add(f.cfg.budget)

	var errs []error
	for _, model := range f.models {
		stream, err, fatal := f.tryModelStream(ctx, deadline, model, req)
		if err == nil {
			return stream, nil
		}
		errs = append(errs, err)
		if fatal || !time.Now().Before(deadline) || ctx.Err() != nil {
			break
		}
	}
	return nil, errors.Join(errs...)
}

// tryModelStream retries only the CONNECTION step (the model.Stream call
// itself) — once it returns a channel successfully, that channel is handed
// straight back to the caller with no further involvement from Fallback.
// Same fatal/retry/skip classification as tryModelGenerate.
func (f *fallbackModel) tryModelStream(ctx context.Context, deadline time.Time, model Model, req GenerateRequest) (<-chan StreamEvent, error, bool) {
	for attempt := 0; ; attempt++ {
		stream, err := model.Stream(ctx, req)
		if err == nil {
			return stream, nil, false
		}

		var aisdkErr *Error
		if !errors.As(err, &aisdkErr) {
			return nil, err, true
		}
		if !aisdkErr.Retryable {
			return nil, err, true
		}
		if aisdkErr.Code == ErrorCodeOverloaded {
			recordFallbackAttempt(ctx, attempt, "skip_overloaded")
			return nil, err, false
		}
		if attempt >= f.cfg.maxRetries {
			recordFallbackAttempt(ctx, attempt, "retries_exhausted")
			return nil, err, false
		}

		wait := f.cfg.backoff(attempt)
		if aisdkErr.RetryAfter > 0 {
			wait = aisdkErr.RetryAfter
		}
		if wait > time.Until(deadline) {
			recordFallbackAttempt(ctx, attempt, "budget_exhausted")
			return nil, err, false
		}

		recordFallbackAttempt(ctx, attempt, "retry")
		select {
		case <-ctx.Done():
			return nil, err, true
		case <-time.After(wait):
		}
	}
}
