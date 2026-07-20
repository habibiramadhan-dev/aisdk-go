# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in aisdk-go, please report it privately rather than opening a public GitHub issue — this gives time to fix the issue before it's publicly disclosed.

Report vulnerabilities via [GitHub's private vulnerability reporting](https://github.com/habibiramadhan-dev/aisdk-go/security/advisories/new) on this repository, or by emailing the maintainer directly at habibiramadhan.work@gmail.com.

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any relevant code, logs, or configuration (with API keys/credentials redacted)

## Supported Versions

As a pre-1.0 project, only the latest released version receives security fixes. Once v1.0.0 ships, this policy will be updated to reflect a supported-version window.

## Design-Level Security Notes

aisdk-go's design already accounts for several trust-boundary concerns relevant to anyone auditing or extending it. The most load-bearing ones:

- **API keys** are only ever supplied via typed constructor options or environment variables the caller reads — never hardcoded, logged, or accepted as request-body fields.
- **Errors are sanitized before wrapping** — `aisdk.Error.Cause` never carries raw HTTP request/response bodies or headers from the underlying provider SDK, so logging an error can't accidentally leak credentials or full prompt content.
- **Tool-call output is always untrusted** — `ToolCall.Arguments` is model-generated output, not validated input. The SDK never executes tool calls or interprets arguments on the caller's behalf.
- **OTel content capture is opt-in, off by default** — `otel.Wrap` never exports full prompt/response text to a tracing backend unless the caller explicitly passes `otel.WithCaptureContent()`.
