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

# Wait for Penny to go idle before touching the simulator (#3171): Xcode and
# the simulator compete with her local model server for the GPU. Before any
# simulator lock or xcodebuild, so a waiting run holds nothing.
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wait-for-penny-idle.sh"

# Picks the iPhone destination on the NEWEST installed iOS runtime, not
# whichever one `-showdestinations` happens to list first (#3150): this box
# carries several simulator runtimes side by side, and the first-listed one
# was an older iOS whose `.searchToolbarBehavior(.minimize)` doesn't collapse
# a `DefaultToolbarItem(kind: .search)` into an icon at all — it falls back
# to an always-visible field instead (works, just not the icon the suite
# asserts), so the scripted run picked a simulator that could never pass the
# search checks even though the fix is correct on the iOS version the app
# ships on. "Newest installed" stays generic on purpose — never hardcode a
# version here, since the next Xcode install changes what "newest" means.
UDID=$(xcodebuild -project penny-client/PennyClient.xcodeproj -scheme PennyClientUITests -showdestinations 2>&1 \
    | /usr/bin/python3 -c "
import re
import sys

best = None
for line in sys.stdin:
    if 'platform:iOS Simulator' not in line:
        continue
    id_match = re.search(r'id:([0-9A-Fa-f-]+)', line)
    name_match = re.search(r'name:([^,}]+)', line)
    os_match = re.search(r'OS:([0-9.]+)', line)
    if not (id_match and name_match and os_match):
        continue
    device_id = id_match.group(1)
    name = name_match.group(1).strip()
    if device_id.startswith('dvtdevice-') or not name.startswith('iPhone'):
        continue
    os_version = tuple(int(part) for part in os_match.group(1).split('.'))
    if best is None or os_version > best[0]:
        best = (os_version, device_id)
if best:
    print(best[1])
")
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
