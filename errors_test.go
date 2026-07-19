package aisdk

import (
	"errors"
	"strings"
	"testing"
)

func TestError_Error_IncludesProviderAndCode(t *testing.T) {
	err := &Error{
		Provider:  "anthropic",
		Code:      ErrorCodeRateLimited,
		Retryable: true,
		Cause:     errors.New("429 too many requests"),
	}

	msg := err.Error()

	if !strings.Contains(msg, "anthropic") {
		t.Errorf("Error() = %q, want it to mention provider %q", msg, "anthropic")
	}
	if !strings.Contains(msg, string(ErrorCodeRateLimited)) {
		t.Errorf("Error() = %q, want it to mention code %q", msg, ErrorCodeRateLimited)
	}
}

func TestError_Unwrap_ReturnsCause(t *testing.T) {
	cause := errors.New("underlying failure")
	err := &Error{Provider: "anthropic", Code: ErrorCodeServerError, Cause: cause}

	if got := errors.Unwrap(err); got != cause {
		t.Errorf("errors.Unwrap(err) = %v, want %v", got, cause)
	}
}
