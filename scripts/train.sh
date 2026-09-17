#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK_ROOT="$PROJECT_ROOT/framework/amp_mjlab/AMP_mjlab"

cd "$FRAMEWORK_ROOT"
exec "${PYTHON:-python}" scripts/train.py Lens110-AMP-GetUp "$@"
