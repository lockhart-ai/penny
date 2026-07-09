#!/bin/bash
# Build the iOS client and run PennyClientTests on a freshly booted simulator.
# Used by `make client-check` locally and by .github/workflows/client-check.yml.
#
# A fresh erase + boot + bootstatus wait per run is load-bearing: launching the
# app on a simulator that is mid-shutdown or reused across back-to-back runs
# fails with FBSOpenApplicationServiceErrorDomain "failed preflight checks".
set -euo pipefail

BUILD_INFO_DERIVED_DATA=""
UDID=$(xcodebuild -project penny-client/PennyClient.xcodeproj -scheme PennyClient -showdestinations 2>/dev/null \
    | awk -F'[{},]' '
        /platform:iOS Simulator/ && /name:iPhone/ {
            for (i = 1; i <= NF; i++) {
                gsub(/^ +| +$/, "", $i)
                if ($i ~ /^id:/) {
                    sub(/^id:/, "", $i)
                    if ($i !~ /^dvtdevice-/) {
                        print $i
                        exit
                    }
                }
            }
        }
    ')
if [ -z "$UDID" ]; then
    echo "client-check: no available iPhone simulator found (is Xcode + an iOS runtime installed?)" >&2
    exit 1
fi

echo "client-check: using simulator $UDID"
xcrun simctl shutdown all >/dev/null 2>&1 || true
xcrun simctl erase "$UDID"
xcrun simctl boot "$UDID"
xcrun simctl bootstatus "$UDID" -b
trap 'xcrun simctl shutdown "$UDID" >/dev/null 2>&1 || true; if [ -n "$BUILD_INFO_DERIVED_DATA" ]; then rm -rf "$BUILD_INFO_DERIVED_DATA"; fi' EXIT

xcodebuild test \
    -project penny-client/PennyClient.xcodeproj \
    -scheme PennyClient \
    -destination "id=$UDID" \
    -skipMacroValidation \
    -skipPackagePluginValidation

BUILD_INFO_DERIVED_DATA="$(mktemp -d)"
EXPECTED_COMMIT_HASH="VERIFYHASH1234"
xcodebuild build \
    -project penny-client/PennyClient.xcodeproj \
    -scheme PennyTestflight \
    -configuration Debug \
    -destination 'generic/platform=iOS Simulator' \
    -derivedDataPath "$BUILD_INFO_DERIVED_DATA" \
    -skipMacroValidation \
    -skipPackagePluginValidation \
    CODE_SIGNING_ALLOWED=NO \
    GIT_COMMIT="$EXPECTED_COMMIT_HASH"

APP_INFO_PLIST="$BUILD_INFO_DERIVED_DATA/Build/Products/Debug-iphonesimulator/PennyTestflight.app/Info.plist"
ACTUAL_COMMIT_HASH="$(/usr/libexec/PlistBuddy -c 'Print :PennyBuildCommitHash' "$APP_INFO_PLIST")"
if [ "$ACTUAL_COMMIT_HASH" != "$EXPECTED_COMMIT_HASH" ]; then
    echo "client-check: PennyBuildCommitHash missing from built app Info.plist" >&2
    echo "client-check: expected $EXPECTED_COMMIT_HASH, got ${ACTUAL_COMMIT_HASH:-<empty>}" >&2
    exit 1
fi
