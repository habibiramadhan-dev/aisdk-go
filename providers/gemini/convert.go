// providers/gemini/convert.go
package gemini

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/habibiramadhan-dev/aisdk-go"
	genaisdk "google.golang.org/genai"
)

// toTools converts aisdk.Tool declarations into Gemini's FunctionDeclaration
// format. Unlike Anthropic (splits the schema into properties/required) or
// OpenAI (unmarshals into a map), Gemini's ParametersJsonSchema field takes
// any value directly — json.RawMessage implements json.Marshaler, so the
// caller's schema marshals through byte-for-byte with no conversion needed.
func toTools(tools []aisdk.Tool) []*genaisdk.Tool {
	if len(tools) == 0 {
		return nil
	}
	declarations := make([]*genaisdk.FunctionDeclaration, 0, len(tools))
	for _, t := range tools {
		declarations = append(declarations, &genaisdk.FunctionDeclaration{
			Name:                 t.Name,
			Description:          t.Description,
			ParametersJsonSchema: t.Parameters,
		})
	}
	return []*genaisdk.Tool{{FunctionDeclarations: declarations}}
}

// findToolCallName recovers the Name of a prior ToolCall by its ID, scanning
// the full message history. Needed because aisdk.ToolResult only carries
// ToolCallID — but Gemini's FunctionResponse.Name is required and must match
// the original FunctionDeclaration/FunctionCall name, and the Gemini API
// itself never gives tool calls a real ID to correlate by (confirmed by
// reading the real API's recorded output — functionCall objects have no "id"
// key). Returns "" if no match is found (best-effort; a caller passing a
// ToolCallID that doesn't exist in history is a caller bug, not something
// this adapter can recover from more gracefully than sending an empty name).
func findToolCallName(messages []aisdk.Message, toolCallID string) string {
	for _, msg := range messages {
		for _, part := range msg.Parts {
			if part.Type == aisdk.ContentPartTypeToolCall && part.ToolCall.ID == toolCallID {
				return part.ToolCall.Name
			}
		}
	}
	return ""
}

func toGenerateContentParams(req aisdk.GenerateRequest) ([]*genaisdk.Content, *genaisdk.GenerateContentConfig) {
	contents := make([]*genaisdk.Content, 0, len(req.Messages))
	for _, msg := range req.Messages {
		var text string
		var parts []*genaisdk.Part
		for _, part := range msg.Parts {
			switch part.Type {
			case aisdk.ContentPartTypeText:
				text += part.Text
			case aisdk.ContentPartTypeToolCall:
				// Errors are ignored (best-effort) — this is model-generated JSON
				// from an earlier turn being replayed, already trusted at
				// generation time, the same trust boundary design.md §8 applies
				// to ToolCall.Arguments generally.
				var args map[string]any
				_ = json.Unmarshal(part.ToolCall.Arguments, &args)
				parts = append(parts, genaisdk.NewPartFromFunctionCall(part.ToolCall.Name, args))
			case aisdk.ContentPartTypeToolResult:
				response := map[string]any{"output": part.ToolResult.Content}
				if part.ToolResult.IsError {
					response = map[string]any{"error": part.ToolResult.Content}
				}
				name := findToolCallName(req.Messages, part.ToolResult.ToolCallID)
				parts = append(parts, genaisdk.NewPartFromFunctionResponse(name, response))
			}
		}
		if text != "" {
			parts = append([]*genaisdk.Part{genaisdk.NewPartFromText(text)}, parts...)
		}
		if len(parts) == 0 {
			continue
		}

		// genaisdk.RoleUser/RoleModel are untyped string constants, so a plain
		// `:=` here would infer role as a bare string (not genai.Role) and fail
		// to compile — the type must be stated explicitly.
		var role genaisdk.Role = genaisdk.RoleUser
		if msg.Role == aisdk.RoleAssistant {
			role = genaisdk.RoleModel
		}
		// NewContentFromParts(parts, role), not a &genaisdk.Content{Role: role, ...}
		// struct literal: Content.Role is a plain string field, and role here is
		// the distinct named type genaisdk.Role — Go's assignability rules reject
		// a named-to-named struct-literal field assignment even though the two
		// share an underlying type, so a direct literal fails to compile. Passing
		// role as a function ARGUMENT typed Role (as this constructor, and
		// config.SystemInstruction's NewContentFromText below, both do) works fine.
		contents = append(contents, genaisdk.NewContentFromParts(parts, role))
	}

	config := &genaisdk.GenerateContentConfig{
		MaxOutputTokens: int32(req.MaxTokens),
	}
	if req.System != "" {
		config.SystemInstruction = genaisdk.NewContentFromText(req.System, genaisdk.RoleUser)
	}
	if req.Temperature != 0 {
		temp := float32(req.Temperature)
		config.Temperature = &temp
	}
	config.Tools = toTools(req.Tools)
	if req.ResponseSchema != nil {
		config.ResponseMIMEType = "application/json"
		config.ResponseJsonSchema = req.ResponseSchema
	}
	return contents, config
}

func toGenerateResponse(resp *genaisdk.GenerateContentResponse) (aisdk.GenerateResponse, error) {
	if len(resp.Candidates) == 0 {
		return aisdk.GenerateResponse{}, fmt.Errorf("gemini: response had no candidates")
	}

	var usage aisdk.Usage
	if resp.UsageMetadata != nil {
		usage = aisdk.Usage{
			InputTokens:  int(resp.UsageMetadata.PromptTokenCount),
			OutputTokens: int(resp.UsageMetadata.CandidatesTokenCount),
		}
	}

	candidate := resp.Candidates[0]
	var parts []aisdk.ContentPart
	hasToolCall := false
	if candidate.Content != nil {
		for i, part := range candidate.Content.Parts {
			if part.FunctionCall != nil {
				hasToolCall = true
				argsJSON, err := json.Marshal(part.FunctionCall.Args)
				if err != nil {
					return aisdk.GenerateResponse{}, fmt.Errorf("gemini: marshaling function call args: %w", err)
				}
				parts = append(parts, aisdk.ContentPart{
					Type: aisdk.ContentPartTypeToolCall,
					ToolCall: &aisdk.ToolCall{
						ID:        fmt.Sprintf("gemini-tool-call-%d", i),
						Name:      part.FunctionCall.Name,
						Arguments: argsJSON,
					},
				})
				continue
			}
			if part.Text != "" {
				parts = append(parts, aisdk.TextPart(part.Text))
			}
		}
	}

	finishReason := toFinishReason(candidate.FinishReason)
	if hasToolCall {
		finishReason = aisdk.FinishReasonToolCalls
	}

	return aisdk.GenerateResponse{
		Message: aisdk.Message{
			Role:  aisdk.RoleAssistant,
			Parts: parts,
		},
		FinishReason: finishReason,
		Usage:        usage,
	}, nil
}

// toFinishReason collapses Gemini's ~16-value FinishReason enum into aisdk's
// 4-value set. Everything besides STOP/MAX_TOKENS (SAFETY, RECITATION,
// PROHIBITED_CONTENT, and the other content-policy variants) maps to Refusal:
// aisdk.FinishReason's own doc comment says a Refusal is "a successful
// response, not an error," which matches a content-policy stop. This
// function never returns FinishReasonToolCalls itself — Gemini's raw
// finishReason stays "STOP" even for a function-call turn, so toGenerateResponse
// overrides the result of this function to ToolCalls after it's already
// iterating Content.Parts to extract tool calls, rather than duplicating
// that Parts scan inside this function too.
func toFinishReason(reason genaisdk.FinishReason) aisdk.FinishReason {
	switch reason {
	case genaisdk.FinishReasonMaxTokens:
		return aisdk.FinishReasonMaxTokens
	case genaisdk.FinishReasonStop, "":
		return aisdk.FinishReasonStop
	default:
		return aisdk.FinishReasonRefusal
	}
}

// mapError wraps the structured genaisdk.APIError directly as Cause, unlike
// the Anthropic/OpenAI adapters which sanitize via errors.New(apiErr.Error()).
// That's not an inconsistency to "fix" — genaisdk.APIError is a plain value
// type (Code/Message/Status/Details) that never embeds *http.Request or
// *http.Response, so there's nothing in it to leak. Wrapping the value
// directly is both safe and strictly more useful: callers can recover the
// structured fields via errors.As instead of just a flattened string.
func mapError(err error) error {
	var apiErr genaisdk.APIError
	if !errors.As(err, &apiErr) {
		return err
	}

	code, retryable := aisdk.ErrorCodeServerError, true
	switch apiErr.Code {
	case 400, 404:
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
		Provider:  "gemini",
		Code:      code,
		Retryable: retryable,
		Cause:     apiErr,
	}
}

func toStreamEvent(chunk *genaisdk.GenerateContentResponse) (aisdk.StreamEvent, bool) {
	if len(chunk.Candidates) == 0 {
		return aisdk.StreamEvent{}, false
	}
	text := chunk.Text()
	if text == "" {
		return aisdk.StreamEvent{}, false
	}
	return aisdk.StreamEvent{Type: aisdk.StreamEventTypeTextDelta, Delta: text}, true
}
