// providers/gemini/gemini_test.go
package gemini_test

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

	"github.com/habibiramadhan-dev/aisdk-go"
	"github.com/habibiramadhan-dev/aisdk-go/aisdktest"
	geminiprovider "github.com/habibiramadhan-dev/aisdk-go/providers/gemini"
	genaisdk "google.golang.org/genai"
)

func TestNew_ModelReturnsNonNilModel(t *testing.T) {
	provider, err := geminiprovider.New(context.Background(), "test-api-key")
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	var model aisdk.Model = provider.Model("gemini-2.0-flash")
	if model == nil {
		t.Fatal("Model() returned nil")
	}
}

func fakeGeminiServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server
}

func newTestProvider(t *testing.T, server *httptest.Server) *geminiprovider.Provider {
	t.Helper()
	provider, err := geminiprovider.New(context.Background(), "test-api-key", func(cc *genaisdk.ClientConfig) {
		cc.HTTPOptions = genaisdk.HTTPOptions{BaseURL: server.URL}
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return provider
}

const fakeSuccessResponse = `{
  "candidates": [{
    "content": {"role": "model", "parts": [{"text": "Hello!"}]},
    "finishReason": "STOP",
    "index": 0
  }],
  "usageMetadata": {
    "promptTokenCount": 10,
    "candidatesTokenCount": 5,
    "totalTokenCount": 15
  },
  "modelVersion": "gemini-2.0-flash"
}`

func TestModel_Generate_ReturnsTextResponse(t *testing.T) {
	server := fakeGeminiServer(t, http.StatusOK, fakeSuccessResponse)
	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

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

func TestModel_Generate_SendsSystemPromptAsSystemInstruction(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fakeSuccessResponse))
	}))
	t.Cleanup(server.Close)

	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		System:    "You are a terse assistant.",
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	sysInstr, ok := capturedBody["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("request body systemInstruction = %+v, want an object", capturedBody["systemInstruction"])
	}
	parts, ok := sysInstr["parts"].([]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("systemInstruction.parts = %+v, want a 1-element array", sysInstr["parts"])
	}
	firstPart, ok := parts[0].(map[string]any)
	if !ok || firstPart["text"] != "You are a terse assistant." {
		t.Errorf("systemInstruction.parts[0] = %+v, want text %q", parts[0], "You are a terse assistant.")
	}

	contents, ok := capturedBody["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("request body contents = %+v, want a 1-element array (the user message only, not the system prompt)", capturedBody["contents"])
	}
}

const fakeAPIKey = "secret-key-should-not-leak"
const fakeRateLimitErrorBody = `{"error":{"code":429,"message":"Rate limit exceeded","status":"RESOURCE_EXHAUSTED"}}`
const fakeAuthErrorBody = `{"error":{"code":401,"message":"Invalid API key","status":"UNAUTHENTICATED"}}`

func TestModel_Generate_MapsRateLimitError(t *testing.T) {
	server := fakeGeminiServer(t, http.StatusTooManyRequests, fakeRateLimitErrorBody)
	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

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
	if aisdkErr.Provider != "gemini" {
		t.Errorf("aisdkErr.Provider = %q, want %q", aisdkErr.Provider, "gemini")
	}
}

func TestModel_Generate_MapsAuthError_NotRetryable(t *testing.T) {
	server := fakeGeminiServer(t, http.StatusUnauthorized, fakeAuthErrorBody)
	provider, err := geminiprovider.New(context.Background(), fakeAPIKey, func(cc *genaisdk.ClientConfig) {
		cc.HTTPOptions = genaisdk.HTTPOptions{BaseURL: server.URL}
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	model := provider.Model("gemini-2.0-flash")

	_, genErr := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
	})

	var aisdkErr *aisdk.Error
	if !errors.As(genErr, &aisdkErr) {
		t.Fatalf("Generate error = %v, want it to unwrap to *aisdk.Error", genErr)
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
		receivedKey = r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(fakeAuthErrorBody))
	}))
	t.Cleanup(server.Close)

	provider, err := geminiprovider.New(context.Background(), fakeAPIKey, func(cc *genaisdk.ClientConfig) {
		cc.HTTPOptions = genaisdk.HTTPOptions{BaseURL: server.URL}
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	model := provider.Model("gemini-2.0-flash")

	_, genErr := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("hi")}}},
	})

	if receivedKey != fakeAPIKey {
		t.Fatalf("server received x-goog-api-key = %q, want %q — test setup is broken", receivedKey, fakeAPIKey)
	}

	fullErrText := fmt.Sprintf("%v", genErr)
	if strings.Contains(fullErrText, fakeAPIKey) {
		t.Fatalf("error string leaked the API key: %s", fullErrText)
	}
}

func fakeGeminiSSEServer(t *testing.T, sseBody string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseBody))
	}))
	t.Cleanup(server.Close)
	return server
}

const fakeStreamSSE = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]},"index":0}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"index":0}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}

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
	server := fakeGeminiSSEServer(t, fakeStreamSSE)
	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

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
	server := fakeGeminiSSEServer(t, fakeStreamSSE)
	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

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

func TestModel_Stream_EmitsErrorEventOnStreamFailure(t *testing.T) {
	// Plain httptest.NewServer, not fakeGeminiSSEServer: Gemini's SDK only
	// produces a typed APIError from the outer HTTP status, checked once
	// before any SSE line is read — an error embedded in a 200-status SSE
	// body is never specially recognized.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":{"code":500,"message":"The server had an error processing your request","status":"INTERNAL"}}`))
	}))
	t.Cleanup(server.Close)

	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

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

	var aisdkErr *aisdk.Error
	if !errors.As(events[0].Err, &aisdkErr) {
		t.Fatalf("events[0].Err = %v, want it to unwrap to *aisdk.Error", events[0].Err)
	}
	if aisdkErr.Code != aisdk.ErrorCodeServerError {
		t.Errorf("aisdkErr.Code = %q, want %q", aisdkErr.Code, aisdk.ErrorCodeServerError)
	}
	if aisdkErr.Provider != "gemini" {
		t.Errorf("aisdkErr.Provider = %q, want %q", aisdkErr.Provider, "gemini")
	}
}

func TestModel_Stream_StopsSendingAfterContextCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		chunks := []string{
			`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"one"}]},"index":0}]}` + "\n\n",
			`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"two"}]},"index":0}]}` + "\n\n",
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

	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

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

func TestModel_Generate_SendsToolDeclarations(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fakeSuccessResponse))
	}))
	t.Cleanup(server.Close)

	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages: []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("What's the weather in Paris?")}}},
		Tools: []aisdk.Tool{{
			Name:        "get_weather",
			Description: "Gets the current weather for a location",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	tools, ok := capturedBody["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("request body tools = %+v, want a 1-element array", capturedBody["tools"])
	}
	tool, ok := tools[0].(map[string]any)
	if !ok {
		t.Fatalf("tools[0] = %+v, want an object", tools[0])
	}
	declarations, ok := tool["functionDeclarations"].([]any)
	if !ok || len(declarations) != 1 {
		t.Fatalf("tools[0].functionDeclarations = %+v, want a 1-element array", tool["functionDeclarations"])
	}
	declaration, ok := declarations[0].(map[string]any)
	if !ok || declaration["name"] != "get_weather" {
		t.Fatalf("functionDeclarations[0] = %+v, want name %q", declarations[0], "get_weather")
	}
	schema, ok := declaration["parametersJsonSchema"].(map[string]any)
	if !ok {
		t.Fatalf("functionDeclarations[0].parametersJsonSchema = %+v, want an object", declaration["parametersJsonSchema"])
	}
	if _, ok := schema["properties"].(map[string]any)["location"]; !ok {
		t.Errorf("parametersJsonSchema.properties = %+v, want a %q key", schema["properties"], "location")
	}
}

const fakeToolCallResponse = `{
  "candidates": [{
    "content": {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "Paris"}}}]},
    "finishReason": "STOP",
    "index": 0
  }],
  "usageMetadata": {
    "promptTokenCount": 20,
    "candidatesTokenCount": 10,
    "totalTokenCount": 30
  },
  "modelVersion": "gemini-2.0-flash"
}`

func TestModel_Generate_ReturnsToolCall(t *testing.T) {
	server := fakeGeminiServer(t, http.StatusOK, fakeToolCallResponse)
	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

	resp, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("What's the weather in Paris?")}}},
		Tools:     []aisdk.Tool{{Name: "get_weather", Description: "...", Parameters: json.RawMessage(`{"type":"object"}`)}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	if len(resp.Message.Parts) != 1 {
		t.Fatalf("resp.Message.Parts = %+v, want exactly 1 part", resp.Message.Parts)
	}
	part := resp.Message.Parts[0]
	if part.Type != aisdk.ContentPartTypeToolCall {
		t.Fatalf("resp.Message.Parts[0].Type = %q, want %q", part.Type, aisdk.ContentPartTypeToolCall)
	}
	if part.ToolCall.ID == "" {
		t.Error("part.ToolCall.ID is empty, want a synthesized ID")
	}
	if part.ToolCall.Name != "get_weather" {
		t.Errorf("part.ToolCall.Name = %q, want %q", part.ToolCall.Name, "get_weather")
	}

	var args map[string]any
	if err := json.Unmarshal(part.ToolCall.Arguments, &args); err != nil {
		t.Fatalf("part.ToolCall.Arguments = %s, not valid JSON: %v", part.ToolCall.Arguments, err)
	}
	if args["location"] != "Paris" {
		t.Errorf("part.ToolCall.Arguments location = %v, want %q", args["location"], "Paris")
	}

	// Gemini's raw finishReason is "STOP" even for a function-call turn — the
	// adapter must override to ToolCalls based on Parts content, not trust
	// the raw enum (confirmed against real recorded Gemini API output).
	if resp.FinishReason != aisdk.FinishReasonToolCalls {
		t.Errorf("resp.FinishReason = %q, want %q (overridden from the raw \"STOP\")", resp.FinishReason, aisdk.FinishReasonToolCalls)
	}
}

func TestGeminiModel_ConformanceSuite(t *testing.T) {
	aisdktest.RunConformanceSuite(t, func(t *testing.T) aisdk.Model {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			tools, _ := body["tools"].([]any)
			generationConfig, _ := body["generationConfig"].(map[string]any)
			hasResponseSchema := len(generationConfig) > 0 && generationConfig["responseJsonSchema"] != nil

			if strings.Contains(r.URL.Path, "streamGenerateContent") {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				if len(tools) > 0 {
					w.Write([]byte(fakeToolCallStreamSSE))
					return
				}
				w.Write([]byte(fakeStreamSSE))
				return
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			if len(tools) > 0 {
				w.Write([]byte(fakeToolCallResponse))
				return
			}
			if hasResponseSchema {
				w.Write([]byte(fakeStructuredOutputResponse))
				return
			}
			w.Write([]byte(fakeSuccessResponse))
		}))
		t.Cleanup(server.Close)

		provider := newTestProvider(t, server)
		return provider.Model("gemini-2.0-flash")
	})
}

const fakeStructuredOutputResponse = `{
  "candidates": [{
    "content": {"role": "model", "parts": [{"text": "{\"city\":\"Paris\",\"temperature_c\":18.5}"}]},
    "finishReason": "STOP",
    "index": 0
  }],
  "usageMetadata": {
    "promptTokenCount": 10,
    "candidatesTokenCount": 5,
    "totalTokenCount": 15
  },
  "modelVersion": "gemini-2.0-flash"
}`

func TestModel_Generate_SendsToolCallAndResultHistory(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fakeSuccessResponse))
	}))
	t.Cleanup(server.Close)

	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages: []aisdk.Message{
			{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("What's the weather in Paris?")}},
			{Role: aisdk.RoleAssistant, Parts: []aisdk.ContentPart{{
				Type:     aisdk.ContentPartTypeToolCall,
				ToolCall: &aisdk.ToolCall{ID: "gemini-tool-call-0", Name: "get_weather", Arguments: json.RawMessage(`{"location":"Paris"}`)},
			}}},
			{Role: aisdk.RoleTool, Parts: []aisdk.ContentPart{{
				Type:       aisdk.ContentPartTypeToolResult,
				ToolResult: &aisdk.ToolResult{ToolCallID: "gemini-tool-call-0", Content: "18°C, cloudy"},
			}}},
		},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	contents, ok := capturedBody["contents"].([]any)
	if !ok || len(contents) != 3 {
		t.Fatalf("request body contents = %+v, want a 3-element array", capturedBody["contents"])
	}

	assistantContent, ok := contents[1].(map[string]any)
	if !ok || assistantContent["role"] != "model" {
		t.Fatalf("contents[1] = %+v, want role %q", contents[1], "model")
	}
	assistantParts, ok := assistantContent["parts"].([]any)
	if !ok || len(assistantParts) != 1 {
		t.Fatalf("contents[1].parts = %+v, want a 1-element array", assistantContent["parts"])
	}
	functionCall, ok := assistantParts[0].(map[string]any)["functionCall"].(map[string]any)
	if !ok || functionCall["name"] != "get_weather" {
		t.Fatalf("contents[1].parts[0].functionCall = %+v, want name %q", assistantParts[0], "get_weather")
	}

	toolResultContent, ok := contents[2].(map[string]any)
	if !ok || toolResultContent["role"] != "user" {
		t.Fatalf("contents[2] = %+v, want role %q", contents[2], "user")
	}
	toolResultParts, ok := toolResultContent["parts"].([]any)
	if !ok || len(toolResultParts) != 1 {
		t.Fatalf("contents[2].parts = %+v, want a 1-element array", toolResultContent["parts"])
	}
	functionResponse, ok := toolResultParts[0].(map[string]any)["functionResponse"].(map[string]any)
	if !ok {
		t.Fatalf("contents[2].parts[0].functionResponse = %+v, want an object", toolResultParts[0])
	}
	// The critical assertion: Name must be recovered from the earlier
	// ToolCall in history (gemini-tool-call-0 → "get_weather"), since
	// aisdk.ToolResult itself only carries ToolCallID, never a Name.
	if functionResponse["name"] != "get_weather" {
		t.Errorf("functionResponse.name = %v, want %q (recovered from message history by ToolCallID)", functionResponse["name"], "get_weather")
	}
	response, ok := functionResponse["response"].(map[string]any)
	if !ok || response["output"] != "18°C, cloudy" {
		t.Errorf("functionResponse.response = %+v, want output %q", functionResponse["response"], "18°C, cloudy")
	}
}

const fakeToolCallStreamSSE = `data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris"}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":10,"totalTokenCount":30}}

`

func TestModel_Stream_EmitsToolCallDeltas(t *testing.T) {
	server := fakeGeminiSSEServer(t, fakeToolCallStreamSSE)
	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

	stream, err := model.Stream(context.Background(), aisdk.GenerateRequest{
		Messages:  []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("What's the weather in Paris?")}}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	events := collectStreamEvents(t, stream)

	var toolCallEvents []aisdk.StreamEvent
	for _, e := range events {
		if e.Type == aisdk.StreamEventTypeToolCallDelta {
			toolCallEvents = append(toolCallEvents, e)
		}
	}
	if len(toolCallEvents) != 1 {
		t.Fatalf("got %d ToolCallDelta events, want exactly 1 (Gemini's function calls arrive atomic, not incremental)", len(toolCallEvents))
	}

	event := toolCallEvents[0]
	if event.ToolCall == nil || event.ToolCall.ID == "" {
		t.Fatal("event.ToolCall is nil or has an empty ID")
	}
	if event.ToolCall.Name != "get_weather" {
		t.Errorf("event.ToolCall.Name = %q, want %q", event.ToolCall.Name, "get_weather")
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(event.Delta), &args); err != nil {
		t.Fatalf("event.Delta = %s, not valid JSON: %v", event.Delta, err)
	}
	if args["location"] != "Paris" {
		t.Errorf("event.Delta location = %v, want %q", args["location"], "Paris")
	}

	var finish *aisdk.StreamEvent
	for i := range events {
		if events[i].Type == aisdk.StreamEventTypeFinish {
			finish = &events[i]
		}
	}
	if finish == nil {
		t.Fatal("no Finish event")
	}
	if finish.FinishReason != aisdk.FinishReasonToolCalls {
		t.Errorf("finish.FinishReason = %q, want %q (overridden from the raw \"STOP\")", finish.FinishReason, aisdk.FinishReasonToolCalls)
	}
}

func TestModel_Generate_SendsResponseSchemaAsResponseJsonSchema(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fakeSuccessResponse))
	}))
	t.Cleanup(server.Close)

	provider := newTestProvider(t, server)
	model := provider.Model("gemini-2.0-flash")

	_, err := model.Generate(context.Background(), aisdk.GenerateRequest{
		Messages:       []aisdk.Message{{Role: aisdk.RoleUser, Parts: []aisdk.ContentPart{aisdk.TextPart("Describe Paris.")}}},
		ResponseSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
		MaxTokens:      64,
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	// responseMimeType/responseJsonSchema are generation-config fields — the
	// genai SDK nests the whole GenerateContentConfig under "generationConfig"
	// in the wire body (unlike "tools" or "systemInstruction", which are
	// top-level), confirmed by inspecting the captured request body directly.
	genConfig, ok := capturedBody["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("request body generationConfig = %+v, want an object", capturedBody["generationConfig"])
	}
	if genConfig["responseMimeType"] != "application/json" {
		t.Errorf("generationConfig.responseMimeType = %v, want %q", genConfig["responseMimeType"], "application/json")
	}
	schema, ok := genConfig["responseJsonSchema"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig.responseJsonSchema = %+v, want an object", genConfig["responseJsonSchema"])
	}
	if _, ok := schema["properties"].(map[string]any)["city"]; !ok {
		t.Errorf("responseJsonSchema.properties = %+v, want a %q key", schema["properties"], "city")
	}
}
