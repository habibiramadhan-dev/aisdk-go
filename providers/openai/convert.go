// providers/openai/convert.go
package openai

import (
	"errors"
	"fmt"

	openaisdk "github.com/openai/openai-go/v2"
	"github.com/habibiramadhan-dev/aisdk-go"
)

func toChatCompletionParams(modelName string, req aisdk.GenerateRequest) openaisdk.ChatCompletionNewParams {
	messages := make([]openaisdk.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)

	if req.System != "" {
		messages = append(messages, openaisdk.SystemMessage(req.System))
	}

	for _, msg := range req.Messages {
		var text string
		for _, part := range msg.Parts {
			if part.Type == aisdk.ContentPartTypeText {
				text += part.Text
			}
		}
		switch msg.Role {
		case aisdk.RoleUser:
			messages = append(messages, openaisdk.UserMessage(text))
		case aisdk.RoleAssistant:
			messages = append(messages, openaisdk.AssistantMessage(text))
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
	return params
}

func toGenerateResponse(resp *openaisdk.ChatCompletion) (aisdk.GenerateResponse, error) {
	if len(resp.Choices) == 0 {
		return aisdk.GenerateResponse{}, fmt.Errorf("openai: response had no choices (id=%s)", resp.ID)
	}
	choice := resp.Choices[0]
	return aisdk.GenerateResponse{
		Message: aisdk.Message{
			Role:  aisdk.RoleAssistant,
			Parts: []aisdk.ContentPart{aisdk.TextPart(choice.Message.Content)},
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

	return &aisdk.Error{
		Provider:  "openai",
		Code:      code,
		Retryable: retryable,
		Cause:     errors.New(apiErr.Error()),
	}
}
