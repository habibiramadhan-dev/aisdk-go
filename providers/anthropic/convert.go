// providers/anthropic/convert.go
package anthropic

import (
	"encoding/json"
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
	params.Tools = toToolUnionParams(req.Tools)

	for _, msg := range req.Messages {
		blocks := make([]anthropicsdk.ContentBlockParamUnion, 0, len(msg.Parts))
		for _, part := range msg.Parts {
			switch part.Type {
			case aisdk.ContentPartTypeText:
				blocks = append(blocks, anthropicsdk.NewTextBlock(part.Text))
			case aisdk.ContentPartTypeToolCall:
				blocks = append(blocks, anthropicsdk.NewToolUseBlock(part.ToolCall.ID, part.ToolCall.Arguments, part.ToolCall.Name))
			case aisdk.ContentPartTypeToolResult:
				blocks = append(blocks, anthropicsdk.NewToolResultBlock(part.ToolResult.ToolCallID, part.ToolResult.Content, part.ToolResult.IsError))
			}
		}
		switch msg.Role {
		case aisdk.RoleUser, aisdk.RoleTool:
			// Tool results are a user-role turn in Anthropic's API — a
			// ToolResultBlockParam is just another content block inside a
			// normal user message, there's no separate "tool" role.
			params.Messages = append(params.Messages, anthropicsdk.NewUserMessage(blocks...))
		case aisdk.RoleAssistant:
			params.Messages = append(params.Messages, anthropicsdk.NewAssistantMessage(blocks...))
		}
	}

	return params
}

func toToolUnionParams(tools []aisdk.Tool) []anthropicsdk.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}
	result := make([]anthropicsdk.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		// Anthropic's InputSchema wants properties/required as separate
		// fields, not a whole JSON Schema document — pluck just those two
		// keys out of the caller-supplied schema via an anonymous struct
		// rather than hand-walking a map[string]any.
		var schema struct {
			Properties any      `json:"properties"`
			Required   []string `json:"required"`
		}
		// Errors are ignored (best-effort) — t.Parameters is caller-supplied
		// and schema validation is the caller's responsibility, the same
		// trust boundary design.md §8 already applies to ToolCall.Arguments.
		json.Unmarshal(t.Parameters, &schema)

		result = append(result, anthropicsdk.ToolUnionParam{
			OfTool: &anthropicsdk.ToolParam{
				Name:        t.Name,
				Description: anthropicsdk.String(t.Description),
				InputSchema: anthropicsdk.ToolInputSchemaParam{
					Properties: schema.Properties,
					Required:   schema.Required,
				},
			},
		})
	}
	return result
}

func toGenerateResponse(msg *anthropicsdk.Message) aisdk.GenerateResponse {
	parts := make([]aisdk.ContentPart, 0, len(msg.Content))
	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			parts = append(parts, aisdk.TextPart(block.Text))
		case "tool_use":
			parts = append(parts, aisdk.ContentPart{
				Type: aisdk.ContentPartTypeToolCall,
				ToolCall: &aisdk.ToolCall{
					ID:        block.ID,
					Name:      block.Name,
					Arguments: block.Input,
				},
			})
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

// toStreamEvent converts a single per-delta SDK stream event into an
// aisdk.StreamEvent. It only handles text/thinking content_block_delta and
// stays a pure, stateless function on purpose — tool-call streaming needs to
// correlate content_block_start's {ID, Name} with later input_json_delta
// fragments by Index, which is inherently stateful, so that's handled
// directly in Stream()'s goroutine locals instead (alongside the similarly
// stateful finishReason/usage accumulation), not here. message_start/
// content_block_stop truly carry nothing we need. message_delta/message_stop
// are also handled statefully in Stream itself, since the terminal Finish
// event needs data from both.
func toStreamEvent(event anthropicsdk.MessageStreamEventUnion) (aisdk.StreamEvent, bool) {
	if event.Type != "content_block_delta" {
		return aisdk.StreamEvent{}, false
	}
	switch event.Delta.Type {
	case "text_delta":
		return aisdk.StreamEvent{Type: aisdk.StreamEventTypeTextDelta, Delta: event.Delta.Text}, true
	case "thinking_delta":
		return aisdk.StreamEvent{Type: aisdk.StreamEventTypeReasoningDelta, Delta: event.Delta.Thinking}, true
	default:
		return aisdk.StreamEvent{}, false
	}
}
