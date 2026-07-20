// internal/httpretry/retryafter_test.go
package httpretry_test

import (
	"net/http"
	"testing"
	"time"

	"github.com/habibiramadhan-dev/aisdk-go/internal/httpretry"
)

func TestParseRetryAfter_IntegerSeconds(t *testing.T) {
	h := http.Header{"Retry-After": []string{"30"}}
	d, ok := httpretry.ParseRetryAfter(h)
	if !ok {
		t.Fatal("ParseRetryAfter returned ok=false, want true")
	}
	if d != 30*time.Second {
		t.Errorf("ParseRetryAfter = %v, want 30s", d)
	}
}

func TestParseRetryAfter_FloatSeconds(t *testing.T) {
	h := http.Header{"Retry-After": []string{"1.5"}}
	d, ok := httpretry.ParseRetryAfter(h)
	if !ok {
		t.Fatal("ParseRetryAfter returned ok=false, want true")
	}
	if d != 1500*time.Millisecond {
		t.Errorf("ParseRetryAfter = %v, want 1.5s", d)
	}
}

func TestParseRetryAfter_HTTPDate(t *testing.T) {
	future := time.Now().Add(45 * time.Second).UTC()
	h := http.Header{"Retry-After": []string{future.Format(http.TimeFormat)}}
	d, ok := httpretry.ParseRetryAfter(h)
	if !ok {
		t.Fatal("ParseRetryAfter returned ok=false, want true")
	}
	// Allow a couple seconds of slack for wall-clock time elapsed during the test.
	if d < 40*time.Second || d > 46*time.Second {
		t.Errorf("ParseRetryAfter = %v, want approximately 45s", d)
	}
}

func TestParseRetryAfter_Absent(t *testing.T) {
	h := http.Header{}
	_, ok := httpretry.ParseRetryAfter(h)
	if ok {
		t.Error("ParseRetryAfter returned ok=true for an absent header, want false")
	}
}

func TestParseRetryAfter_Unparseable(t *testing.T) {
	h := http.Header{"Retry-After": []string{"not-a-valid-value"}}
	_, ok := httpretry.ParseRetryAfter(h)
	if ok {
		t.Error("ParseRetryAfter returned ok=true for garbage input, want false")
	}
}

func TestParseRetryAfter_PastHTTPDate(t *testing.T) {
	past := time.Now().Add(-1 * time.Hour).UTC()
	h := http.Header{"Retry-After": []string{past.Format(http.TimeFormat)}}
	_, ok := httpretry.ParseRetryAfter(h)
	if ok {
		t.Error("ParseRetryAfter returned ok=true for an already-past date, want false")
	}
}
