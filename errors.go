package aisdk

import "fmt"

// ErrorCode classifies an Error independent of which provider produced it.
type ErrorCode string

const (
	ErrorCodeRateLimited      ErrorCode = "rate_limited"
	ErrorCodeAuthFailed       ErrorCode = "auth_failed"
	ErrorCodePermissionDenied ErrorCode = "permission_denied"
	ErrorCodeInvalidRequest   ErrorCode = "invalid_request"
	ErrorCodeOverloaded       ErrorCode = "overloaded"
	ErrorCodeServerError      ErrorCode = "server_error"
	ErrorCodeTimeout          ErrorCode = "timeout"
	ErrorCodeContextLength    ErrorCode = "context_length"
)

// Error is the common error type every provider adapter maps its native SDK
// errors into. Cause is sanitized by the adapter before wrapping — it must never
// carry raw HTTP headers or request/response bodies (see design.md §8).
type Error struct {
	Provider  string
	Code      ErrorCode
	Retryable bool
	RequestID string
	Cause     error
}

func (e *Error) Error() string {
	return fmt.Sprintf("aisdk: %s: %s (retryable=%v): %v", e.Provider, e.Code, e.Retryable, e.Cause)
}

func (e *Error) Unwrap() error {
	return e.Cause
}
