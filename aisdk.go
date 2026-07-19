// aisdk.go
package aisdk

import (
	"context"
	"encoding/json"
)

// Provider produces Models by name (e.g. "claude-sonnet-5", "gpt-5").
type Provider interface {
	Model(name string) Model
}

// Model is the unified interface every provider adapter implements.
type Model interface {
	Generate(ctx context.Context, req GenerateRequest) (GenerateResponse, error)
	Stream(ctx context.Context, req GenerateRequest) (<-chan StreamEvent, error)
}

// Role identifies who sent a Message. There is no "system" role — see
// GenerateRequest.System.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentPartType discriminates the variants of ContentPart.
type ContentPartType string

const (
	ContentPartTypeText       ContentPartType = "text"
	ContentPartTypeReasoning  ContentPartType = "reasoning"
	ContentPartTypeToolCall   ContentPartType = "tool_call"
	ContentPartTypeToolResult ContentPartType = "tool_result"
)

// ContentPart is one piece of a Message's content.
type ContentPart struct {
	Type       ContentPartType
	Text       string      // set when Type is Text or Reasoning
	ToolCall   *ToolCall   // set when Type is ToolCall
	ToolResult *ToolResult // set when Type is ToolResult
}

// TextPart is a convenience constructor for the common case of a plain text part.
func TextPart(text string) ContentPart {
	return ContentPart{Type: ContentPartTypeText, Text: text}
}

// Message is one turn in a conversation.
type Message struct {
	Role  Role
	Parts []ContentPart
}

// Tool describes a function the model may call. Parameters is a JSON Schema.
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// ToolCall is model-generated output naming a Tool and its arguments. Arguments
// is untrusted, model-generated data — the SDK never executes it (see design.md §8).
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ToolResult is the caller's response to a ToolCall, sent back in a follow-up
// Message with Role RoleTool.
type ToolResult struct {
	ToolCallID string
	Content    string
	IsError    bool
}

// GenerateRequest is the unified request shape across all providers.
type GenerateRequest struct {
	// System is the system prompt/instruction. Never a Message with a system
	// role — Anthropic and Gemini both expose this as a dedicated top-level
	// field natively; the OpenAI adapter folds it into Messages itself.
	System          string
	Messages        []Message
	Tools           []Tool
	MaxTokens       int
	Temperature     float64
	ProviderOptions map[string]any
}

// FinishReason explains why generation stopped. A Refusal or MaxTokens result
// is a successful response, not an error.
type FinishReason string

const (
	FinishReasonStop      FinishReason = "stop"
	FinishReasonMaxTokens FinishReason = "max_tokens"
	FinishReasonToolCalls FinishReason = "tool_calls"
	FinishReasonRefusal   FinishReason = "refusal"
)

// Usage reports token consumption for a single Generate call.
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// GenerateResponse is the unified response shape across all providers.
type GenerateResponse struct {
	Message      Message
	FinishReason FinishReason
	Usage        Usage
}
