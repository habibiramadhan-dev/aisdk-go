package aisdk

import (
	"context"
	"fmt"
	"strings"
)

// defaultAskMaxTokens is the MaxTokens Ask sends when the caller has no
// reason to think about it — every provider rejects a request with no
// max_tokens set, so the simple path can't just leave it at zero.
const defaultAskMaxTokens = 1024

// Ask wraps Generate for the single-message, no-tools case: send one string,
// get one string back, no GenerateRequest assembly required.
func Ask(ctx context.Context, model Model, prompt string) (string, error) {
	resp, err := model.Generate(ctx, GenerateRequest{
		Messages: []Message{
			{Role: RoleUser, Parts: []ContentPart{TextPart(prompt)}},
		},
		MaxTokens: defaultAskMaxTokens,
	})
	if err != nil {
		return "", fmt.Errorf("aisdk: ask: %w", err)
	}

	var text strings.Builder
	for _, part := range resp.Message.Parts {
		if part.Type == ContentPartTypeText {
			text.WriteString(part.Text)
		}
	}
	return text.String(), nil
}
