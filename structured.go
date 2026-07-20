// structured.go
package aisdk

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/habibiramadhan-dev/aisdk-go/internal/schema"
)

// GenerateStructured reflects T into a JSON Schema (internal/schema),
// constrains model's response to match it, and unmarshals the result into
// T. It calls Model.Generate, never Model.Stream — streaming structured
// output ("streamObject") is out of scope for v1 (design.md §3).
//
// T must not be self-referential — see internal/schema.Reflect's doc
// comment for why (an unrecoverable crash during reflection, not a
// returnable error).
func GenerateStructured[T any](ctx context.Context, model Model, req GenerateRequest) (T, error) {
	var zero T

	rawSchema, err := schema.Reflect[T]()
	if err != nil {
		return zero, fmt.Errorf("aisdk: reflecting schema for structured output: %w", err)
	}
	req.ResponseSchema = rawSchema

	resp, err := model.Generate(ctx, req)
	if err != nil {
		return zero, err
	}

	var text string
	for _, part := range resp.Message.Parts {
		if part.Type == ContentPartTypeText {
			text += part.Text
		}
	}

	var result T
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return zero, fmt.Errorf("aisdk: unmarshaling structured output: %w", err)
	}
	return result, nil
}
