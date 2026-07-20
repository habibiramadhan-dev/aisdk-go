// providers/openai/convert.go
package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	openaisdk "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/habibiramadhan-dev/aisdk-go"
	"github.com/habibiramadhan-dev/aisdk-go/internal/httpretry"
)

func toChatCompletionParams(modelName string, req aisdk.GenerateRequest) openaisdk.ChatCompletionNewParams {
	messages := make([]openaisdk.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)

	if req.System != "" {
		messages = append(messages, openaisdk.SystemMessage(req.System))
	}

	for _, msg := range req.Messages {
		var text string
		var toolCallParts []aisdk.ContentPart
		for _, part := range msg.Parts {
			switch part.Type {
			case aisdk.ContentPartTypeText:
				text += part.Text
			case aisdk.ContentPartTypeToolCall:
				toolCallParts = append(toolCallParts, part)
			}
		}

		switch msg.Role {
		case aisdk.RoleUser:
			messages = append(messages, openaisdk.UserMessage(text))

		case aisdk.RoleAssistant:
			if len(toolCallParts) == 0 {
				messages = append(messages, openaisdk.AssistantMessage(text))
				continue
			}
			// No plain-string helper covers an assistant turn that made tool
			// calls (openaisdk.AssistantMessage only ever sets Content) — the
			// param struct has to be hand-built, matching what a fresh
			// go doc github.com/openai/openai-go/v2.ChatCompletionAssistantMessageParam
			// confirms: Content is "required unless tool_calls ... is specified".
			assistant := openaisdk.ChatCompletionAssistantMessageParam{}
			if text != "" {
				assistant.Content.OfString = param.NewOpt(text)
			}
			for _, part := range toolCallParts {
				assistant.ToolCalls = append(assistant.ToolCalls, openaisdk.ChatCompletionMessageToolCallUnionParam{
					OfFunction: &openaisdk.ChatCompletionMessageFunctionToolCallParam{
						ID: part.ToolCall.ID,
						Function: openaisdk.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name:      part.ToolCall.Name,
							Arguments: string(part.ToolCall.Arguments),
						},
					},
				})
			}
			messages = append(messages, openaisdk.ChatCompletionMessageParamUnion{OfAssistant: &assistant})

		case aisdk.RoleTool:
			// OpenAI has no equivalent of Anthropic's "all results in one
			// message" shape — every tool result is its own role:"tool"
			// message, so one aisdk.Message with N ToolResult parts becomes
			// N entries in the flat OpenAI messages slice.
			for _, part := range msg.Parts {
				if part.Type != aisdk.ContentPartTypeToolResult {
					continue
				}
				messages = append(messages, openaisdk.ToolMessage(part.ToolResult.Content, part.ToolResult.ToolCallID))
			}
		}
	}

	params := openaisdk.ChatCompletionNewParams{
		Model:               modelName,
		Messages:            messages,
		MaxCompletionTokens: openaisdk.Int(int64(req.MaxTokens)),
	}
	if req.Temperature != 0 {
		params.Temperature = openaisdk.Float(req.Temperature)
	}
	params.Tools = toToolUnionParams(req.Tools)
	if req.ResponseSchema != nil {
		params.ResponseFormat = openaisdk.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &openaisdk.ResponseFormatJSONSchemaParam{
				JSONSchema: openaisdk.ResponseFormatJSONSchemaJSONSchemaParam{
					// "response" is a fixed name — aisdk's GenerateStructured
					// has no caller-facing concept of naming the schema, and
					// OpenAI requires *some* name here regardless.
					Name:   "response",
					Schema: req.ResponseSchema,
					Strict: openaisdk.Bool(true),
				},
			},
		}
	}
	return params
}

func toToolUnionParams(tools []aisdk.Tool) []openaisdk.ChatCompletionToolUnionParam {
	if len(tools) == 0 {
		return nil
	}
	result := make([]openaisdk.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		// Errors are ignored (best-effort) — t.Parameters is caller-supplied
		// and schema validation is the caller's responsibility, the same
		// trust boundary design.md §8 already applies to ToolCall.Arguments.
		var parameters openaisdk.FunctionParameters
		json.Unmarshal(t.Parameters, &parameters)

		result = append(result, openaisdk.ChatCompletionFunctionTool(openaisdk.FunctionDefinitionParam{
			Name:        t.Name,
			Description: openaisdk.String(t.Description),
			Parameters:  parameters,
		}))
	}
	return result
}

func toGenerateResponse(resp *openaisdk.ChatCompletion) (aisdk.GenerateResponse, error) {
	if len(resp.Choices) == 0 {
		return aisdk.GenerateResponse{}, fmt.Errorf("openai: response had no choices (id=%s)", resp.ID)
	}
	choice := resp.Choices[0]

	var parts []aisdk.ContentPart
	if choice.Message.Content != "" {
		parts = append(parts, aisdk.TextPart(choice.Message.Content))
	}
	for _, tc := range choice.Message.ToolCalls {
		parts = append(parts, aisdk.ContentPart{
			Type: aisdk.ContentPartTypeToolCall,
			ToolCall: &aisdk.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			},
		})
	}

	return aisdk.GenerateResponse{
		Message: aisdk.Message{
			Role:  aisdk.RoleAssistant,
			Parts: parts,
		},
		FinishReason: toFinishReason(choice.FinishReason),
		Usage: aisdk.Usage{
			InputTokens:  int(resp.Usage.PromptTokens),
			OutputTokens: int(resp.Usage.CompletionTokens),
		},
	}, nil
}

func toFinishReason(reason string) aisdk.FinishReason {
	switch reason {
	case "length":
		return aisdk.FinishReasonMaxTokens
	case "tool_calls", "function_call":
		return aisdk.FinishReasonToolCalls
	case "content_filter":
		return aisdk.FinishReasonRefusal
	default: // "stop"
		return aisdk.FinishReasonStop
	}
}

func toStreamEvent(chunk openaisdk.ChatCompletionChunk) (aisdk.StreamEvent, bool) {
	if len(chunk.Choices) == 0 {
		return aisdk.StreamEvent{}, false
	}
	delta := chunk.Choices[0].Delta
	if delta.Content == "" {
		return aisdk.StreamEvent{}, false
	}
	return aisdk.StreamEvent{Type: aisdk.StreamEventTypeTextDelta, Delta: delta.Content}, true
}

func mapError(err error) error {
	var apiErr *openaisdk.Error
	if !errors.As(err, &apiErr) {
		return err
	}

	// Default covers 5xx and any status this switch doesn't name explicitly —
	// treating unlisted/future statuses as a retryable server error is the
	// safer default rather than silently refusing to retry something that
	// might recover.
	code, retryable := aisdk.ErrorCodeServerError, true
	switch apiErr.StatusCode {
	case 400, 404:
		// 404 groups with 400 here (both InvalidRequest, not retryable):
		// retrying an identical request against a resource that doesn't
		// exist (e.g. a bad model name) will fail the same way every time.
		code, retryable = aisdk.ErrorCodeInvalidRequest, false
	case 401:
		code, retryable = aisdk.ErrorCodeAuthFailed, false
	case 403:
		code, retryable = aisdk.ErrorCodePermissionDenied, false
	case 429:
		code, retryable = aisdk.ErrorCodeRateLimited, true
	case 503:
		code, retryable = aisdk.ErrorCodeOverloaded, true
	}

	// apiErr.Response is read for its header before Cause is built from the
	// sanitized apiErr.Error() string below — reading .Response.Header here
	// does not reintroduce the leak errors.New(apiErr.Error()) exists to
	// prevent, since only the specific Retry-After value is extracted, never
	// the full request/response object itself. apiErr.Response can
	// theoretically be nil (per SDK doc comments), hence the guard.
	var retryAfter time.Duration
	if apiErr.Response != nil {
		retryAfter, _ = httpretry.ParseRetryAfter(apiErr.Response.Header)
	}

	return &aisdk.Error{
		Provider:   "openai",
		Code:       code,
		Retryable:  retryable,
		RetryAfter: retryAfter,
		Cause:      errors.New(apiErr.Error()),
	}
}
