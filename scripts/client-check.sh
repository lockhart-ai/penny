#!/bin/bash
# Build the iOS client and run PennyClientTests on a freshly booted simulator.
# Used by `make client-check` locally and by .github/workflows/client-check.yml.
#
# A fresh erase + boot + bootstatus wait per run is load-bearing: launching the
# app on a simulator that is mid-shutdown or reused across back-to-back runs
# fails with FBSOpenApplicationServiceErrorDomain "failed preflight checks".
set -euo pipefail

UDID=$(xcrun simctl list devices available \
    | grep -m1 -E '^ *iPhone' \
    | grep -oE '[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}')
if [ -z "$UDID" ]; then
    echo "client-check: no available iPhone simulator found (is Xcode + an iOS runtime installed?)" >&2
    exit 1
fi

echo "client-check: using simulator $UDID"
xcrun simctl shutdown all >/dev/null 2>&1 || true
xcrun simctl erase "$UDID"
xcrun simctl boot "$UDID"
xcrun simctl bootstatus "$UDID" -b
trap 'xcrun simctl shutdown "$UDID" >/dev/null 2>&1 || true' EXIT

xcodebuild test \
    -project penny-client/PennyClient.xcodeproj \
    -scheme PennyClient \
    -destination "id=$UDID" \
    -skipMacroValidation \
    -skipPackagePluginValidation
