package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v2/option"
	"github.com/habibiramadhan-dev/aisdk-go"
	"github.com/habibiramadhan-dev/aisdk-go/aisdktest"
	openaiprovider "github.com/habibiramadhan-dev/aisdk-go/providers/openai"
)

func TestNew_ModelReturnsNonNilModel(t *testing.T) {
	provider := openaiprovider.New("test-api-key")

	var model aisdk.Model = provider.Model("gpt-4o")
	if model == nil {
		t.Fatal("Model() returned nil")
	}
}

func fakeOpenAIServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server
}

const fakeSuccessResponse = `{
  "id": "chatcmpl_test123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!", "refusal": null},
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}`

func TestModel_Generate_ReturnsTextResponse(t *testing.T) {
	server := fakeOpenAIServer(t, http.StatusOK, fakeSuccessResponse)

	provider := openaiprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	resp, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	if len(resp.Message.Parts) != 1 || resp.Message.Parts[0].Text != "Hello!" {
		t.Errorf("resp.Message.Parts = %+v, want a single text part %q", resp.Message.Parts, "Hello!")
	}
	if resp.FinishReason != aisdk.FinishReasonStop {
		t.Errorf("resp.FinishReason = %q, want %q", resp.FinishReason, aisdk.FinishReasonStop)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("resp.Usage = %+v, want {10 5}", resp.Usage)
	}
}

func TestModel_Generate_SendsSystemPromptAsFirstMessage(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fakeSuccessResponse))
	}))
	t.Cleanup(server.Close)

	provider := openaiprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		System:    "You are a terse assistant.",
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	messages, ok := capturedBody["messages"].([]any)
	if !ok || len(messages) != 2 {
		t.Fatalf("request body messages = %+v, want a 2-element array (system, user)", capturedBody["messages"])
	}
	firstMsg, ok := messages[0].(map[string]any)
	if !ok {
		t.Fatalf("messages[0] = %+v, want an object", messages[0])
	}
	if firstMsg["role"] != "system" {
		t.Errorf("messages[0].role = %v, want %q", firstMsg["role"], "system")
	}
	if firstMsg["content"] != "You are a terse assistant." {
		t.Errorf("messages[0].content = %v, want %q", firstMsg["content"], "You are a terse assistant.")
	}
}

const fakeAPIKey = "secret-key-should-not-leak"
const fakeRateLimitErrorBody = `{"error":{"message":"Rate limit exceeded","type":"requests","code":"rate_limit_exceeded"}}`
const fakeAuthErrorBody = `{"error":{"message":"Invalid API key","type":"invalid_request_error","code":"invalid_api_key"}}`

func TestModel_Generate_MapsRateLimitError(t *testing.T) {
	server := fakeOpenAIServer(t, http.StatusTooManyRequests, fakeRateLimitErrorBody)
	provider := openaiprovider.New(fakeAPIKey, option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
	})

	var aisdkErr *aisdk.Error
	if !errors.As(err, &aisdkErr) {
		t.Fatalf("Generate error = %v, want it to unwrap to *aisdk.Error", err)
	}
	if aisdkErr.Code != aisdk.ErrorCodeRateLimited {
		t.Errorf("aisdkErr.Code = %q, want %q", aisdkErr.Code, aisdk.ErrorCodeRateLimited)
	}
	if !aisdkErr.Retryable {
		t.Error("aisdkErr.Retryable = false, want true for HTTP 429")
	}
	if aisdkErr.Provider != "openai" {
		t.Errorf("aisdkErr.Provider = %q, want %q", aisdkErr.Provider, "openai")
	}
}

func TestModel_Generate_MapsAuthError_NotRetryable(t *testing.T) {
	server := fakeOpenAIServer(t, http.StatusUnauthorized, fakeAuthErrorBody)
	provider := openaiprovider.New(fakeAPIKey, option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
	})

	var aisdkErr *aisdk.Error
	if !errors.As(err, &aisdkErr) {
		t.Fatalf("Generate error = %v, want it to unwrap to *aisdk.Error", err)
	}
	if aisdkErr.Code != aisdk.ErrorCodeAuthFailed {
		t.Errorf("aisdkErr.Code = %q, want %q", aisdkErr.Code, aisdk.ErrorCodeAuthFailed)
	}
	if aisdkErr.Retryable {
		t.Error("aisdkErr.Retryable = true, want false for HTTP 401")
	}
}

func TestModel_Generate_ErrorNeverLeaksAPIKey(t *testing.T) {
	var receivedKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedKey = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(fakeAuthErrorBody))
	}))
	t.Cleanup(server.Close)

	provider := openaiprovider.New(fakeAPIKey, option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
	})

	if receivedKey != fakeAPIKey {
		t.Fatalf("server received Authorization bearer = %q, want %q — test setup is broken", receivedKey, fakeAPIKey)
	}

	fullErrText := fmt.Sprintf("%v", err)
	if strings.Contains(fullErrText, fakeAPIKey) {
		t.Fatalf("error string leaked the API key: %s", fullErrText)
	}
}

func fakeOpenAISSEServer(t *testing.T, sseBody string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseBody))
	}))
	t.Cleanup(server.Close)
	return server
}

const fakeStreamSSE = `data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}

data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"lo!"},"finish_reason":null}]}

data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

data: [DONE]

`

func collectStreamEvents(t *testing.T, events <-chan aisdk.StreamEvent) []aisdk.StreamEvent {
	t.Helper()
	var collected []aisdk.StreamEvent
	for e := range events {
		collected = append(collected, e)
	}
	return collected
}

func TestModel_Stream_ReturnsTextDeltas(t *testing.T) {
	server := fakeOpenAISSEServer(t, fakeStreamSSE)
	provider := openaiprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	events := collectStreamEvents(t, stream)

	var text string
	for _, e := range events {
		if e.Type == aisdk.StreamEventTypeTextDelta {
			text += e.Delta
		}
	}
	if text != "Hello!" {
		t.Errorf("concatenated text deltas = %q, want %q", text, "Hello!")
	}
}

func TestModel_Stream_RequestsIncludeUsage(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fakeStreamSSE))
	}))
	t.Cleanup(server.Close)

	provider := openaiprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	collectStreamEvents(t, stream)

	streamOptions, ok := capturedBody["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("request body stream_options = %+v, want an object", capturedBody["stream_options"])
	}
	if streamOptions["include_usage"] != true {
		t.Errorf("stream_options.include_usage = %v, want true — without this, OpenAI never sends usage on any chunk", streamOptions["include_usage"])
	}
}

func TestModel_Stream_EmitsFinishWithUsageAndReason(t *testing.T) {
	server := fakeOpenAISSEServer(t, fakeStreamSSE)
	provider := openaiprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	events := collectStreamEvents(t, stream)

	var finishEvents []aisdk.StreamEvent
	for _, e := range events {
		if e.Type == aisdk.StreamEventTypeFinish {
			finishEvents = append(finishEvents, e)
		}
	}
	if len(finishEvents) != 1 {
		t.Fatalf("got %d Finish events, want exactly 1", len(finishEvents))
	}

	finish := finishEvents[0]
	if finish.FinishReason != aisdk.FinishReasonStop {
		t.Errorf("finish.FinishReason = %q, want %q", finish.FinishReason, aisdk.FinishReasonStop)
	}
	if finish.Usage.InputTokens != 10 || finish.Usage.OutputTokens != 5 {
		t.Errorf("finish.Usage = %+v, want {10 5} — if this is {0 0}, IncludeUsage isn't actually being sent on the request", finish.Usage)
	}

	if events[len(events)-1].Type != aisdk.StreamEventTypeFinish {
		t.Errorf("last event type = %q, want Finish to be the terminal event", events[len(events)-1].Type)
	}
}

const fakeStreamErrorSSE = `data: {"error":{"message":"The server had an error processing your request","type":"server_error","code":null}}

`

func TestModel_Stream_EmitsErrorEventOnStreamFailure(t *testing.T) {
	server := fakeOpenAISSEServer(t, fakeStreamErrorSSE)
	provider := openaiprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	events := collectStreamEvents(t, stream)

	if len(events) != 1 {
		t.Fatalf("got %d events, want exactly 1 (the error event)", len(events))
	}
	if events[0].Type != aisdk.StreamEventTypeError {
		t.Fatalf("events[0].Type = %q, want %q", events[0].Type, aisdk.StreamEventTypeError)
	}
	if events[0].Err == nil {
		t.Fatal("events[0].Err = nil, want a non-nil error")
	}
	// Unlike a Generate()-path error (backed by a real HTTP status and
	// *openaisdk.Error), OpenAI's stream decoder reports in-stream "error" SSE
	// frames as a plain formatted string, so mapError's errors.As check
	// doesn't match here and falls back to returning the error unchanged.
	// This event does NOT unwrap to *aisdk.Error — that's expected.
	if !strings.Contains(events[0].Err.Error(), "error") {
		t.Errorf("events[0].Err = %v, want it to mention the stream error", events[0].Err)
	}
}

func TestModel_Stream_StopsSendingAfterContextCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		chunks := []string{
			`data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"one"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"two"},"finish_reason":null}]}` + "\n\n",
		}
		for _, c := range chunks {
			w.Write([]byte(c))
			if flusher != nil {
				flusher.Flush()
			}
			select {
			case <-time.After(50 * time.Millisecond):
			case <-r.Context().Done():
				return
			}
		}
	}))
	t.Cleanup(server.Close)

	provider := openaiprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("gpt-4o")

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := model.Stream(ctx, aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	<-stream
	cancel()

	done := make(chan struct{})
	go func() {
		for range stream {
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("stream channel was not closed within 2s of context cancellation — goroutine leak")
	}
}

func TestOpenAIModel_ConformanceSuite(t *testing.T) {
	aisdktest.RunConformanceSuite(t, func(t *testing.T) aisdk.Model {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusInternalServerError)
				w.Write([]byte(`{"error":{"message":"fake server: failed to decode request body","type":"server_error"}}`))
				return
			}

			if streaming, _ := body["stream"].(bool); streaming {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				w.Write([]byte(fakeStreamSSE))
				return
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(fakeSuccessResponse))
		}))
		t.Cleanup(server.Close)

		provider := openaiprovider.New("test-api-key", option.WithBaseURL(server.URL))
		return provider.Model("gpt-4o")
	})
}
