// providers/gemini/convert.go
package gemini

import (
	"errors"
	"fmt"

	"github.com/habibiramadhan-dev/aisdk-go"
	genaisdk "google.golang.org/genai"
)

func toGenerateContentParams(req aisdk.GenerateRequest) ([]*genaisdk.Content, *genaisdk.GenerateContentConfig) {
	contents := make([]*genaisdk.Content, 0, len(req.Messages))
	for _, msg := range req.Messages {
		var text string
		for _, part := range msg.Parts {
			if part.Type == aisdk.ContentPartTypeText {
				text += part.Text
			}
		}
		var role genaisdk.Role = genaisdk.RoleUser
		if msg.Role == aisdk.RoleAssistant {
			role = genaisdk.RoleModel
		}
		contents = append(contents, genaisdk.NewContentFromText(text, role))
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

	return aisdk.GenerateResponse{
		Message: aisdk.Message{
			Role:  aisdk.RoleAssistant,
			Parts: []aisdk.ContentPart{aisdk.TextPart(resp.Text())},
		},
		FinishReason: toFinishReason(resp.Candidates[0].FinishReason),
		Usage:        usage,
	}, nil
}

// toFinishReason collapses Gemini's ~16-value FinishReason enum into aisdk's
// 4-value set. Everything besides STOP/MAX_TOKENS (SAFETY, RECITATION,
// PROHIBITED_CONTENT, and the other content-policy variants) maps to Refusal:
// aisdk.FinishReason's own doc comment says a Refusal is "a successful
// response, not an error," which matches a content-policy stop. Detecting
// FinishReasonToolCalls would require inspecting Candidate.Content.Parts for
// FunctionCall parts — out of scope until Fase 4 (tool-calling).
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
