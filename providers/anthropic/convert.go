// providers/anthropic/convert.go
package anthropic

import (
	"errors"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/habibiramadhan-dev/aisdk-go"
)

func toMessageNewParams(modelName string, req aisdk.GenerateRequest) anthropicsdk.MessageNewParams {
	params := anthropicsdk.MessageNewParams{
		Model:     modelName,
		MaxTokens: int64(req.MaxTokens),
		Messages:  make([]anthropicsdk.MessageParam, 0, len(req.Messages)),
	}

	if req.System != "" {
		params.System = []anthropicsdk.TextBlockParam{{Text: req.System}}
	}
	if req.Temperature != 0 {
		params.Temperature = anthropicsdk.Float(req.Temperature)
	}

	for _, msg := range req.Messages {
		blocks := make([]anthropicsdk.ContentBlockParamUnion, 0, len(msg.Parts))
		for _, part := range msg.Parts {
			if part.Type == aisdk.ContentPartTypeText {
				blocks = append(blocks, anthropicsdk.NewTextBlock(part.Text))
			}
		}
		switch msg.Role {
		case aisdk.RoleUser:
			params.Messages = append(params.Messages, anthropicsdk.NewUserMessage(blocks...))
		case aisdk.RoleAssistant:
			params.Messages = append(params.Messages, anthropicsdk.NewAssistantMessage(blocks...))
		}
	}

	return params
}

func toGenerateResponse(msg *anthropicsdk.Message) aisdk.GenerateResponse {
	parts := make([]aisdk.ContentPart, 0, len(msg.Content))
	for _, block := range msg.Content {
		if block.Type == "text" {
			parts = append(parts, aisdk.TextPart(block.Text))
		}
	}

	return aisdk.GenerateResponse{
		Message: aisdk.Message{
			Role:  aisdk.RoleAssistant,
			Parts: parts,
		},
		FinishReason: toFinishReason(msg.StopReason),
		Usage: aisdk.Usage{
			InputTokens:  int(msg.Usage.InputTokens),
			OutputTokens: int(msg.Usage.OutputTokens),
		},
	}
}

func toFinishReason(reason anthropicsdk.StopReason) aisdk.FinishReason {
	switch reason {
	case anthropicsdk.StopReasonMaxTokens:
		return aisdk.FinishReasonMaxTokens
	case anthropicsdk.StopReasonToolUse:
		return aisdk.FinishReasonToolCalls
	case anthropicsdk.StopReasonRefusal:
		return aisdk.FinishReasonRefusal
	default: // end_turn, stop_sequence, pause_turn
		return aisdk.FinishReasonStop
	}
}

func mapError(err error) error {
	var apiErr *anthropicsdk.Error
	if !errors.As(err, &apiErr) {
		return err
	}

	// Default covers api_error (opaque server-side failure, worth retrying) and any
	// ErrorType this switch doesn't name explicitly — new ErrorTypes the SDK adds
	// later land here too, so treating them as a retryable server error is the safer
	// default rather than silently refusing to retry something that might recover.
	code, retryable := aisdk.ErrorCodeServerError, true
	switch apiErr.Type() {
	case anthropicsdk.ErrorTypeInvalidRequestError:
		code, retryable = aisdk.ErrorCodeInvalidRequest, false
	case anthropicsdk.ErrorTypeNotFoundError:
		// Retrying an identical request against a resource that doesn't exist
		// (e.g. a bad model name) will fail the same way every time.
		code, retryable = aisdk.ErrorCodeInvalidRequest, false
	case anthropicsdk.ErrorTypeAuthenticationError:
		code, retryable = aisdk.ErrorCodeAuthFailed, false
	case anthropicsdk.ErrorTypePermissionError:
		code, retryable = aisdk.ErrorCodePermissionDenied, false
	case anthropicsdk.ErrorTypeRateLimitError:
		code, retryable = aisdk.ErrorCodeRateLimited, true
	case anthropicsdk.ErrorTypeTimeoutError:
		code, retryable = aisdk.ErrorCodeTimeout, true
	case anthropicsdk.ErrorTypeOverloadedError:
		code, retryable = aisdk.ErrorCodeOverloaded, true
	case anthropicsdk.ErrorTypeBillingError:
		// aisdk.ErrorCode has no billing-specific value yet; PermissionDenied is the
		// closest existing code ("you can't do this until something changes account-side").
		code, retryable = aisdk.ErrorCodePermissionDenied, false
	}

	return &aisdk.Error{
		Provider:  "anthropic",
		Code:      code,
		Retryable: retryable,
		RequestID: apiErr.RequestID,
		Cause:     errors.New(apiErr.Error()),
	}
}
