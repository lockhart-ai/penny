# penny-client
iOS client for the penny personal assistant

Build and test from the repo root with `make client-check` (requires Xcode; runs
the standalone service tests, app-hosted `PennyClientTests`, and the Testflight
build verification on a freshly booted simulator). For service-only changes,
`make client-services-check` runs `PennyServicesStandaloneTests` without
building or launching an app target. CI runs the complete gate on pull requests
touching `penny-client/`.

Settings → Image Attachments controls automatic images for the entire iOS channel:
cited-page images, same-site images, and broader related images. All switches start
enabled. Changes save immediately to the server with confirmation and error state;
the connection settings' Save button is separate. Explicit attachments and images
generated for the current request always remain deliverable. The controls affect
new replies; existing messages and the History attachment option are unchanged.
An older server that does not expose all four switches shows disabled controls.
