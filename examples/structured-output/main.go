// examples/structured-output/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/habibiramadhan-dev/aisdk-go"
	anthropicprovider "github.com/habibiramadhan-dev/aisdk-go/providers/anthropic"
)

// MovieReview is the shape we want the model's response constrained to. No
// field has `omitempty` — every field is always present in the model's
// output (see design.md §3 for why this phase doesn't lower `required` for
// omitempty fields).
type MovieReview struct {
	Title   string `json:"title"`
	Rating  int    `json:"rating"`
	Summary string `json:"summary"`
}

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("set ANTHROPIC_API_KEY to run this example")
	}

	provider := anthropicprovider.New(apiKey)
	model := provider.Model("claude-sonnet-5")
	ctx := context.Background()

	// GenerateStructured reflects MovieReview into a JSON Schema, constrains
	// the model's response to match it, and unmarshals the result into a
	// MovieReview for us — no manual json.Unmarshal at the call site.
	review, err := aisdk.GenerateStructured[MovieReview](ctx, model, aisdk.GenerateRequest{
		Messages: []aisdk.Message{
			{
				Role:  aisdk.RoleUser,
				Parts: []aisdk.ContentPart{aisdk.TextPart("Write a short review of the movie 'The Matrix'.")},
			},
		},
		MaxTokens: 256,
	})
	if err != nil {
		log.Fatalf("GenerateStructured failed: %v", err)
	}

	fmt.Println("Title:", review.Title)
	fmt.Println("Rating:", review.Rating)
	fmt.Println("Summary:", review.Summary)
}
