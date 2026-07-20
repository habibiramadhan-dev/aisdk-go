// otel/otel.go
package otel

import (
	"context"
	"encoding/json"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/habibiramadhan-dev/aisdk-go"
)

// WrapOption configures an otel.Wrap-ed Model.
type WrapOption func(*wrapConfig)

// WithCaptureContent enables recording full prompt/completion text as span
// attributes (gen_ai.prompt / gen_ai.completion). Off by default — per
// design.md §8, consumer applications may put PII or secrets in prompts
// they never intended to export to a tracing backend, so capturing full
// content must be an explicit opt-in, not a default.
func WithCaptureContent() WrapOption {
	return func(c *wrapConfig) { c.captureContent = true }
}

// WithTracerProvider sets the trace.TracerProvider used to create spans.
// Defaults to otel.GetTracerProvider() (the process-global provider,
// resolved once at Wrap time) if not set — pass one explicitly for test
// isolation (e.g. a sdktrace.NewTracerProvider backed by an in-memory
// exporter), so tests don't mutate global state.
func WithTracerProvider(tp trace.TracerProvider) WrapOption {
	return func(c *wrapConfig) { c.tracerProvider = tp }
}

type wrapConfig struct {
	captureContent bool
	tracerProvider trace.TracerProvider
}

// Wrap returns a Model that records an OpenTelemetry span, following GenAI
// semantic conventions, around every Generate/Stream call. See design.md §7.
func Wrap(model aisdk.Model, opts ...WrapOption) aisdk.Model {
	cfg := wrapConfig{tracerProvider: otel.GetTracerProvider()}
	for _, opt := range opts {
		opt(&cfg)
	}
	return &wrappedModel{
		model:  model,
		cfg:    cfg,
		tracer: cfg.tracerProvider.Tracer("github.com/habibiramadhan-dev/aisdk-go/otel"),
	}
}

type wrappedModel struct {
	model  aisdk.Model
	cfg    wrapConfig
	tracer trace.Tracer
}

// modelIdentity returns ok=false when the wrapped Model doesn't implement
// aisdk.ModelInfo (e.g. a test fake, or a Fallback-composed model with no
// single provider/model identity) — otel.Wrap degrades gracefully rather
// than requiring every Model to support introspection.
func (w *wrappedModel) modelIdentity() (provider, modelName string, ok bool) {
	mi, isModelInfo := w.model.(aisdk.ModelInfo)
	if !isModelInfo {
		return "", "", false
	}
	return mi.Provider(), mi.ModelName(), true
}

func spanName(operation, modelName string, haveIdentity bool) string {
	if !haveIdentity {
		return "aisdk." + operation
	}
	return operation + " " + modelName
}

func (w *wrappedModel) startSpan(ctx context.Context, operation string, req aisdk.GenerateRequest) (context.Context, trace.Span) {
	provider, modelName, ok := w.modelIdentity()
	var attrs []attribute.KeyValue
	if ok {
		attrs = append(attrs, attribute.String("gen_ai.system", provider), attribute.String("gen_ai.request.model", modelName))
	}
	if req.MaxTokens > 0 {
		attrs = append(attrs, attribute.Int("gen_ai.request.max_tokens", req.MaxTokens))
	}
	if w.cfg.captureContent {
		if promptJSON, err := json.Marshal(req.Messages); err == nil {
			attrs = append(attrs, attribute.String("gen_ai.prompt", string(promptJSON)))
		}
	}
	return w.tracer.Start(ctx, spanName(operation, modelName, ok), trace.WithAttributes(attrs...))
}

func recordUsage(span trace.Span, usage aisdk.Usage, finishReason aisdk.FinishReason) {
	span.SetAttributes(
		attribute.Int("gen_ai.usage.input_tokens", usage.InputTokens),
		attribute.Int("gen_ai.usage.output_tokens", usage.OutputTokens),
		attribute.String("gen_ai.response.finish_reason", string(finishReason)),
	)
}

func (w *wrappedModel) Generate(ctx context.Context, req aisdk.GenerateRequest) (aisdk.GenerateResponse, error) {
	ctx, span := w.startSpan(ctx, "generate", req)
	defer span.End()

	resp, err := w.model.Generate(ctx, req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return resp, err
	}

	recordUsage(span, resp.Usage, resp.FinishReason)
	if w.cfg.captureContent {
		if completionJSON, err := json.Marshal(resp.Message); err == nil {
			span.SetAttributes(attribute.String("gen_ai.completion", string(completionJSON)))
		}
	}
	span.SetStatus(codes.Ok, "")
	return resp, nil
}

func (w *wrappedModel) Stream(ctx context.Context, req aisdk.GenerateRequest) (<-chan aisdk.StreamEvent, error) {
	ctx, span := w.startSpan(ctx, "stream", req)

	inner, err := w.model.Stream(ctx, req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		span.End()
		return nil, err
	}

	out := make(chan aisdk.StreamEvent)
	go func() {
		defer close(out)
		defer span.End()

		// Unlike Generate's full resp.Message, a stream only has incremental
		// text deltas to accumulate — captured content for a streamed call
		// is plain text, not the full JSON-marshaled Message structure
		// Generate's gen_ai.completion carries (there's no equivalent
		// "final Message" available mid-stream to marshal).
		var completion string

		for event := range inner {
			select {
			case out <- event:
			case <-ctx.Done():
				return
			}
			switch event.Type {
			case aisdk.StreamEventTypeTextDelta:
				if w.cfg.captureContent {
					completion += event.Delta
				}
			case aisdk.StreamEventTypeFinish:
				recordUsage(span, event.Usage, event.FinishReason)
				if w.cfg.captureContent {
					span.SetAttributes(attribute.String("gen_ai.completion", completion))
				}
				span.SetStatus(codes.Ok, "")
			case aisdk.StreamEventTypeError:
				span.RecordError(event.Err)
				span.SetStatus(codes.Error, event.Err.Error())
			}
		}
	}()
	return out, nil
}
