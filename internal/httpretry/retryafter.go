// internal/httpretry/retryafter.go
package httpretry

import (
	"net/http"
	"strconv"
	"time"
)

// ParseRetryAfter parses an HTTP Retry-After header value into a duration
// from now. Returns 0, false if the header is absent, unparseable in either
// legal format, or names a time already in the past.
//
// Retry-After is legally either an integer/float number of seconds, or an
// HTTP-date — confirmed by reading Anthropic's and OpenAI's own (identical,
// both Stainless-generated) internal retry logic, which parses it the same
// dual-format way rather than assuming integer-seconds only.
func ParseRetryAfter(h http.Header) (time.Duration, bool) {
	raw := h.Get("Retry-After")
	if raw == "" {
		return 0, false
	}
	if secs, err := strconv.ParseFloat(raw, 64); err == nil {
		return time.Duration(secs * float64(time.Second)), true
	}
	if t, err := http.ParseTime(raw); err == nil {
		if d := time.Until(t); d > 0 {
			return d, true
		}
	}
	return 0, false
}
