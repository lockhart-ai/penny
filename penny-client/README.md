# penny-client
iOS client for the penny personal assistant

Build and test from the repo root with `make client-check` (requires Xcode; runs
the standalone service tests, app-hosted `PennyClientTests`, and the Testflight
build verification on a freshly booted simulator). For service-only changes,
`make client-services-check` runs `PennyServicesStandaloneTests` without
building or launching an app target. CI runs the complete gate on pull requests
touching `penny-client/`.
