#!/bin/bash
# Build and run PennyServicesStandaloneTests without launching an app target.
set -euo pipefail

UDID=$(xcodebuild -project penny-client/PennyClient.xcodeproj -scheme PennyServicesStandaloneTests -showdestinations 2>&1 \
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
    echo "client-services-check: no available iPhone simulator found" >&2
    exit 1
fi

echo "client-services-check: using simulator $UDID"
xcrun simctl shutdown all >/dev/null 2>&1 || true
xcrun simctl erase "$UDID"
xcrun simctl boot "$UDID"
xcrun simctl bootstatus "$UDID" -b
trap 'xcrun simctl shutdown "$UDID" >/dev/null 2>&1 || true' EXIT

xcodebuild test \
    -project penny-client/PennyClient.xcodeproj \
    -scheme PennyServicesStandaloneTests \
    -destination "id=$UDID" \
    -skipMacroValidation \
    -skipPackagePluginValidation
