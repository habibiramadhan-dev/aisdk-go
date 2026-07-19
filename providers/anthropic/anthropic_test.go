package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/habibiramadhan-dev/aisdk-go"
	"github.com/habibiramadhan-dev/aisdk-go/aisdktest"
	anthropicprovider "github.com/habibiramadhan-dev/aisdk-go/providers/anthropic"
)

func TestNew_ModelReturnsNonNilModel(t *testing.T) {
	provider := anthropicprovider.New("test-api-key")

	var model aisdk.Model = provider.Model("claude-sonnet-5")
	if model == nil {
		t.Fatal("Model() returned nil")
	}
}

func fakeAnthropicServer(t *testing.T, status int, body string) *httptest.Server {
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
  "id": "msg_test123",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-5",
  "content": [{"type": "text", "text": "Hello!"}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 10, "output_tokens": 5}
}`

func TestModel_Generate_ReturnsTextResponse(t *testing.T) {
	server := fakeAnthropicServer(t, http.StatusOK, fakeSuccessResponse)

	provider := anthropicprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("claude-sonnet-5")

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

func TestModel_Generate_SendsSystemPromptAtTopLevel(t *testing.T) {
	var capturedBody map[string]any
	var decodeErr error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decodeErr = json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fakeSuccessResponse))
	}))
	t.Cleanup(server.Close)

	provider := anthropicprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("claude-sonnet-5")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		System:    "You are a terse assistant.",
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if decodeErr != nil {
		t.Fatalf("failed to decode request body: %v", decodeErr)
	}

	systemBlocks, ok := capturedBody["system"].([]any)
	if !ok || len(systemBlocks) != 1 {
		t.Fatalf("request body system = %+v, want a single-element array", capturedBody["system"])
	}
	block := systemBlocks[0].(map[string]any)
	if block["text"] != "You are a terse assistant." {
		t.Errorf("system block text = %v, want %q", block["text"], "You are a terse assistant.")
	}

	messages, ok := capturedBody["messages"].([]any)
	if !ok || len(messages) == 0 {
		t.Fatalf("request body messages = %+v, want a non-empty array", capturedBody["messages"])
	}
	firstMsg, ok := messages[0].(map[string]any)
	if !ok {
		t.Fatalf("messages[0] = %+v, want an object", messages[0])
	}
	if firstMsg["role"] != "user" {
		t.Errorf("messages[0].role = %v, want %q (system must not appear as a message role)", firstMsg["role"], "user")
	}
}

const fakeRateLimitResponse = `{"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}`
const fakeAuthErrorResponse = `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`
const fakeAPIKey = "secret-key-should-not-leak"

func TestModel_Generate_MapsRateLimitError(t *testing.T) {
	server := fakeAnthropicServer(t, http.StatusTooManyRequests, fakeRateLimitResponse)
	provider := anthropicprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("claude-sonnet-5")

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
		t.Error("aisdkErr.Retryable = false, want true for rate_limit_error")
	}
	if aisdkErr.Provider != "anthropic" {
		t.Errorf("aisdkErr.Provider = %q, want %q", aisdkErr.Provider, "anthropic")
	}
}

func TestModel_Generate_MapsAuthError_NotRetryable(t *testing.T) {
	server := fakeAnthropicServer(t, http.StatusUnauthorized, fakeAuthErrorResponse)
	provider := anthropicprovider.New(fakeAPIKey, option.WithBaseURL(server.URL))
	model := provider.Model("claude-sonnet-5")

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
		t.Error("aisdkErr.Retryable = true, want false for authentication_error")
	}
}

func TestModel_Generate_ErrorNeverLeaksAPIKey(t *testing.T) {
	var receivedKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedKey = r.Header.Get("X-Api-Key")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(fakeAuthErrorResponse))
	}))
	t.Cleanup(server.Close)

	provider := anthropicprovider.New(fakeAPIKey, option.WithBaseURL(server.URL))
	model := provider.Model("claude-sonnet-5")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
	})

	// Positive control: confirm the key really was sent, so the negative check
	// below actually proves something rather than passing vacuously.
	if receivedKey != fakeAPIKey {
		t.Fatalf("server received x-api-key = %q, want %q — test setup is broken", receivedKey, fakeAPIKey)
	}

	fullErrText := fmt.Sprintf("%v", err)
	if strings.Contains(fullErrText, fakeAPIKey) {
		t.Fatalf("error string leaked the API key: %s", fullErrText)
	}
}

func TestAnthropicModel_ConformanceSuite(t *testing.T) {
	aisdktest.RunConformanceSuite(t, func(t *testing.T) aisdk.Model {
		server := fakeAnthropicServer(t, http.StatusOK, fakeSuccessResponse)
		provider := anthropicprovider.New("test-api-key", option.WithBaseURL(server.URL))
		return provider.Model("claude-sonnet-5")
	})
}

func fakeAnthropicSSEServer(t *testing.T, sseBody string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseBody))
	}))
	t.Cleanup(server.Close)
	return server
}

const fakeStreamSSE = `event: message_start
data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5,"input_tokens":10}}

event: message_stop
data: {"type":"message_stop"}

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
	server := fakeAnthropicSSEServer(t, fakeStreamSSE)
	provider := anthropicprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("claude-sonnet-5")

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

func TestModel_Stream_EmitsFinishWithUsageAndReason(t *testing.T) {
	server := fakeAnthropicSSEServer(t, fakeStreamSSE)
	provider := anthropicprovider.New("test-api-key", option.WithBaseURL(server.URL))
	model := provider.Model("claude-sonnet-5")

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
		t.Errorf("finish.Usage = %+v, want {10 5}", finish.Usage)
	}

	if events[len(events)-1].Type != aisdk.StreamEventTypeFinish {
		t.Errorf("last event type = %q, want Finish to be the terminal event", events[len(events)-1].Type)
	}
}
