# Contributing to aisdk-go

Thanks for your interest in aisdk-go. This is currently a solo-maintained portfolio project, but issues, bug reports, and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/habibiramadhan-dev/aisdk-go
cd aisdk-go
go build ./...
go test ./...
```

No API keys are needed to build or run the default test suite — every provider adapter is tested against fake HTTP transports (`httptest.Server`), never real provider APIs, so `go test ./...` works offline and free of charge.

## Running the integration test suite

A separate test suite, gated behind the `integration` build tag, hits real provider APIs. It's skipped by default (including in CI's normal test job) and only useful if you have your own API keys:

```bash
export ANTHROPIC_API_KEY=...   # optional — tests for a missing key are skipped
export OPENAI_API_KEY=...      # optional
export GEMINI_API_KEY=...      # optional
go test -tags=integration ./... -v
```

## Code style

- Go standard formatting (`gofmt`) — CI enforces this via `golangci-lint`.
- Minimal comments: explain *why*, not *what* — a comment restating what the next line already says in code is worse than no comment.
- Every package should ship with tests, not just the shared conformance suite.
- Follow the existing pattern for provider adapters if adding one: implement `aisdk.Model` (`Generate`/`Stream`), map provider errors into `aisdk.Error` with sanitized causes (never wrap a raw SDK error that embeds `*http.Request`/`*http.Response`), and pass the shared `aisdktest.RunConformanceSuite`.

## Reporting bugs / requesting features

Open a GitHub issue. For security vulnerabilities specifically, see [SECURITY.md](./SECURITY.md) instead — don't open a public issue for those.

## Pull requests

- Keep PRs focused — one logical change per PR is easier to review than a large mixed one.
- Include tests for new behavior.
- Run `go build ./...`, `go vet ./...`, and `go test ./...` locally before opening a PR — CI runs the same checks plus `golangci-lint` and `govulncheck`.
