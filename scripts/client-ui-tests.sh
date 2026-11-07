#!/bin/bash
# Build PennyDev and run the PennyClientUITests smoke suite (#3106) on a
# freshly booted simulator. Used by `make client-ui-tests`, LOCAL ONLY —
# CI's client-check.yml never runs this (macOS runners are scarce); the
# PennyClientUITests target is deliberately kept out of PennyClient.xctestplan
# and the PennyClient scheme's TestAction, so `make client-check` never
# builds or runs it.
#
# Same fresh erase + boot + bootstatus wait as client-check.sh: launching the
# app on a simulator that is mid-shutdown or reused across back-to-back runs
# fails with FBSOpenApplicationServiceErrorDomain "failed preflight checks".
set -euo pipefail

UDID=$(xcodebuild -project penny-client/PennyClient.xcodeproj -scheme PennyClientUITests -showdestinations 2>&1 \
    | awk -F'[{},]' '
        /platform:iOS Simulator/ && /name:iPhone/ && !found {
            for (i = 1; i <= NF; i++) {
                gsub(/^ +| +$/, "", $i)
                if ($i ~ /^id:/) {
                    sub(/^id:/, "", $i)
                    if ($i !~ /^dvtdevice-/) {
                        print $i
                        found = 1
                    }
                }
            }
        }
    ')
if [ -z "$UDID" ]; then
    echo "client-ui-tests: no available iPhone simulator found (is Xcode + an iOS runtime installed?)" >&2
    exit 1
fi

echo "client-ui-tests: using simulator $UDID"
xcrun simctl shutdown "$UDID" >/dev/null 2>&1 || true
xcrun simctl erase "$UDID"
xcrun simctl boot "$UDID"
xcrun simctl bootstatus "$UDID" -b
trap 'xcrun simctl shutdown "$UDID" >/dev/null 2>&1 || true' EXIT

xcodebuild test \
    -project penny-client/PennyClient.xcodeproj \
    -scheme PennyClientUITests \
    -destination "id=$UDID" \
    -skipMacroValidation \
    -skipPackagePluginValidation
