#!/bin/bash
# Build the iOS client and run PennyClientTests on a freshly booted simulator.
# Used by `make client-check` locally and by .github/workflows/client-check.yml.
#
# A fresh erase + boot + bootstatus wait per run is load-bearing: launching the
# app on a simulator that is mid-shutdown or reused across back-to-back runs
# fails with FBSOpenApplicationServiceErrorDomain "failed preflight checks".
set -euo pipefail

# Wait for Penny to go idle before touching the simulator (#3171): Xcode and
# the simulator compete with her local model server for the GPU. Before any
# simulator lock or xcodebuild, so a waiting run holds nothing.
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wait-for-penny-idle.sh"

echo "client-check: $(xcodebuild -version | head -n 1)"

# Prints "<runtime identifier>\n<runtime version>\n<iPhone device type
# identifier>\n<iPhone device type name>" for the newest available iOS
# runtime the selected Xcode can see that supports at least one iPhone
# device type, or nothing if no such runtime is installed.
#
# A version tie is broken toward the runtime with more supported device
# types: some Xcode betas install a second, same-version runtime pinned to
# one preview device (e.g. an "iOS-27-2-iPhone19-4" runtime supporting only
# "iPhone Duo") alongside the general-purpose runtime for that same iOS
# version — the general one is always the richer of the two.
pick_newest_ios_runtime_and_iphone() {
    xcrun simctl list runtimes -j | python3 -c "
import json, sys

runtimes = json.load(sys.stdin)['runtimes']
ios = [r for r in runtimes if r.get('platform') == 'iOS' and r.get('isAvailable')]
if not ios:
    sys.exit(0)
newest = max(
    ios,
    key=lambda r: (
        tuple(int(p) for p in r['version'].split('.')),
        len(r['supportedDeviceTypes']),
    ),
)
iphones = [d for d in newest['supportedDeviceTypes'] if d.get('productFamily') == 'iPhone']
if not iphones:
    sys.exit(0)
print(newest['identifier'])
print(newest['version'])
print(iphones[0]['identifier'])
print(iphones[0]['name'])
"
}

BUILD_INFO_DERIVED_DATA=""
UDID=$(xcodebuild -project penny-client/PennyClient.xcodeproj -scheme PennyClient -showdestinations 2>&1 \
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

# Most runner images already carry a bootable iPhone simulator device, which
# the pass above finds. Some don't: a fresh runner image can have an iOS
# runtime installed with zero simulator devices ever created, so
# -showdestinations lists none (#3156). Rather than fail the job before any
# test runs, create one deterministically instead of failing: the newest
# installed iOS runtime the selected Xcode can see, and an iPhone device
# type it supports — downloading the platform first if no iOS runtime is
# installed at all.
if [ -z "$UDID" ]; then
    echo "client-check: no iPhone simulator found via -showdestinations; creating one" >&2
    SELECTION="$(pick_newest_ios_runtime_and_iphone)"
    if [ -z "$SELECTION" ]; then
        echo "client-check: no iOS runtime installed; downloading one" >&2
        xcodebuild -downloadPlatform iOS
        SELECTION="$(pick_newest_ios_runtime_and_iphone)"
    fi
    if [ -z "$SELECTION" ]; then
        echo "client-check: no iOS runtime with an iPhone device type available, even after -downloadPlatform" >&2
        exit 1
    fi
    RUNTIME_ID=$(sed -n '1p' <<<"$SELECTION")
    RUNTIME_VERSION=$(sed -n '2p' <<<"$SELECTION")
    DEVICE_TYPE_ID=$(sed -n '3p' <<<"$SELECTION")
    DEVICE_TYPE_NAME=$(sed -n '4p' <<<"$SELECTION")
    echo "client-check: creating simulator — iOS $RUNTIME_VERSION, $DEVICE_TYPE_NAME" >&2
    UDID=$(xcrun simctl create "CI iPhone" "$DEVICE_TYPE_ID" "$RUNTIME_ID")
fi

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
    -scheme PennyServicesStandaloneTests \
    -destination "id=$UDID" \
    -skipMacroValidation \
    -skipPackagePluginValidation

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
