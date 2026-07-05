#!/bin/zsh
set -euo pipefail

repo_root="${CI_WORKSPACE:-$(git rev-parse --show-toplevel)}"
cd "$repo_root/penny-client"

build_number="${CI_BUILD_NUMBER:-$(git rev-list --count HEAD)}"
perl -0pi -e "s/CURRENT_PROJECT_VERSION = [^;]+;/CURRENT_PROJECT_VERSION = $build_number;/g" \
    PennyClient.xcodeproj/project.pbxproj
