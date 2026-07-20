package aisdk

import (
	"fmt"
	"time"
)

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
	// RetryAfter is the provider's requested wait time before retrying, parsed
	// from a Retry-After response header when present. Zero means the
	// provider didn't supply one (or, for Gemini, that its SDK's error type
	// structurally has no access to response headers at all — an honest gap,
	// not a bug; see providers/gemini's mapError, which never sets this).
	RetryAfter time.Duration
	Cause      error
}

func (e *Error) Error() string {
	return fmt.Sprintf("aisdk: %s: %s (retryable=%v): %v", e.Provider, e.Code, e.Retryable, e.Cause)
}

func (e *Error) Unwrap() error {
	return e.Cause
}
