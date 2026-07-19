// stream.go
package aisdk

// StreamEventType discriminates the variants of StreamEvent.
type StreamEventType string

const (
	StreamEventTypeTextDelta      StreamEventType = "text_delta"
	StreamEventTypeReasoningDelta StreamEventType = "reasoning_delta"
	StreamEventTypeToolCallDelta  StreamEventType = "tool_call_delta"
	StreamEventTypeFinish         StreamEventType = "finish"
	StreamEventTypeError          StreamEventType = "error"
)

// StreamEvent is one item from a Model.Stream channel. Defined in Fase 1;
// no adapter emits these yet until Fase 2 (design.md §10).
type StreamEvent struct {
	Type         StreamEventType
	Delta        string
	ToolCall     *ToolCall
	FinishReason FinishReason // set when Type is StreamEventTypeFinish
	Usage        Usage        // set when Type is StreamEventTypeFinish
	Err          error        // set when Type is StreamEventTypeError
}
