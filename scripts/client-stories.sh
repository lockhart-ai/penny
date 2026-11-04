#!/bin/bash
# Build PennyClient (Debug) once and screenshot every story (#2965): each
# story renders one screen from fixture data via `-PennyStory <id>`, with no
# network and no push registration. Used by `make client-stories`.
#
# The story id list has one source of truth: `StoryID.swift`'s `case` lines,
# read below — never a second, hand-kept list in this script.
set -euo pipefail

SIMULATOR_NAME="iPhone 18 Pro"
SIMULATOR_RUNTIME_PREFIX="com.apple.CoreSimulator.SimRuntime.iOS-27-2"
BUNDLE_ID="com.blockoftext.pennyclient"
OUTPUT_DIR="penny-client/build/stories"
STORY_FILE="penny-client/PennyClient/Stories/StoryID.swift"

STORY_IDS=$(grep -oE '^\s*case [a-zA-Z0-9]+' "$STORY_FILE" | awk '{print $2}')
if [ -z "$STORY_IDS" ]; then
    echo "client-stories: no story ids found in $STORY_FILE" >&2
    exit 1
fi

UDID=$(xcrun simctl list devices available -j | /usr/bin/python3 -c "
import json, sys
data = json.load(sys.stdin)
for runtime, devices in data['devices'].items():
    if not runtime.startswith('$SIMULATOR_RUNTIME_PREFIX'):
        continue
    for device in devices:
        if device['name'] == '$SIMULATOR_NAME':
            print(device['udid'])
            sys.exit(0)
")
if [ -z "$UDID" ]; then
    echo "client-stories: no '$SIMULATOR_NAME' simulator found on an iOS 27.2 runtime" >&2
    exit 1
fi

echo "client-stories: using simulator $UDID"
mkdir -p "$OUTPUT_DIR"
# Only this device: never shut down other simulators (other agents may be using them).
xcrun simctl boot "$UDID" 2>/dev/null || true
xcrun simctl bootstatus "$UDID" -b
xcrun simctl ui "$UDID" appearance light
xcrun simctl status_bar "$UDID" override --time 9:41 --batteryState charged --batteryLevel 100 --cellularBars 4 --wifiBars 3
trap 'xcrun simctl status_bar "$UDID" clear >/dev/null 2>&1 || true; xcrun simctl shutdown "$UDID" >/dev/null 2>&1 || true; if [ -n "${DERIVED_DATA:-}" ]; then rm -rf "$DERIVED_DATA"; fi' EXIT

DERIVED_DATA="$(mktemp -d)"
xcodebuild build \
    -project penny-client/PennyClient.xcodeproj \
    -scheme PennyClient \
    -configuration Debug \
    -destination "id=$UDID" \
    -derivedDataPath "$DERIVED_DATA" \
    -skipMacroValidation \
    -skipPackagePluginValidation

APP_PATH="$DERIVED_DATA/Build/Products/Debug-iphonesimulator/PennyDev.app"
xcrun simctl install "$UDID" "$APP_PATH"

for story_id in $STORY_IDS; do
    echo "client-stories: $story_id"
    xcrun simctl launch --terminate-running-process "$UDID" "$BUNDLE_ID" -PennyStory "$story_id" >/dev/null
    sleep 2
    xcrun simctl io "$UDID" screenshot "$OUTPUT_DIR/$story_id.png" >/dev/null
done

echo "client-stories: screenshots saved to $OUTPUT_DIR"
