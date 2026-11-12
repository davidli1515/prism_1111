#!/usr/bin/env bash
set -euo pipefail

GEM5_DIR="/scratch/gem5-TAO"
OUT_DIR="${PWD}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TAR="${OUT_DIR}/gem5-TAO-src_${STAMP}.tar.gz"
OUT_MANIFEST="${OUT_DIR}/gem5-TAO-src_${STAMP}.manifest.txt"
OUT_SYMLINKS="${OUT_DIR}/gem5-TAO-src_${STAMP}.symlinks.txt"

ITEMS=(
  "build_opts"
  "build_tools"
  "configs"
  "docs"
  "ext"
  "include/gem5"
  "site_scons"
  "src"
  "system/arm"
  "tests"
  "util"

  ".clang-format"
  ".git-blame-ignore-revs"
  ".gitignore"
  ".mailmap"
  ".pre-commit-config.yaml"
  "CODE-OF-CONDUCT.md"
  "CONTRIBUTING.md"
  "COPYING"
  "KCONFIG.md"
  "LICENSE"
  "MAINTAINERS.yaml"
  "README.md"
  "RELEASE-NOTES.md"
  "SConstruct"
  "TESTING.md"
  "optional-requirements.txt"
  "pyproject.toml"
  "requirements.txt"
)

if [[ ! -d "$GEM5_DIR" ]]; then
  echo "ERROR: gem5 dir not found: $GEM5_DIR" >&2
  exit 1
fi

# 过滤掉不存在的条目（避免 tar 报错）
EXISTING=()
for it in "${ITEMS[@]}"; do
  if [[ -e "${GEM5_DIR}/${it}" ]]; then
    EXISTING+=("$it")
  else
    echo "WARN: not found, skip: ${it}" >&2
  fi
done

echo "Packing from: $GEM5_DIR"
echo "Output tarball: $OUT_TAR"
echo "NOTE: build/ will be excluded."

# 1) 统计源目录 symlink（验收用）
echo "[1/4] Counting symlinks in source tree..."
( cd "$GEM5_DIR" && find "${EXISTING[@]}" -type l 2>/dev/null | sort ) > "$OUT_SYMLINKS" || true
echo "Symlinks in source: $(wc -l < "$OUT_SYMLINKS")"
echo "Symlink list saved to: $OUT_SYMLINKS"

# 2) 打包（排除 build/；保留权限/时间戳；不要跟随 symlink）
echo "[2/4] Creating tar.gz..."
tar -C "$GEM5_DIR" \
  --exclude='build' \
  --exclude='build/**' \
  --numeric-owner \
  --format=gnu \
  -czf "$OUT_TAR" \
  "${EXISTING[@]}"

# 3) 生成 manifest（列出 tar 内容，含 symlink 标记）
echo "[3/4] Writing manifest..."
tar -tvzf "$OUT_TAR" > "$OUT_MANIFEST"
echo "Manifest saved to: $OUT_MANIFEST"

# 4) 快速检查 tar 里是否包含 symlink（tar 列表里以 'l' 开头）
echo "[4/4] Checking symlinks inside tar..."
SYMS_IN_TAR=$(grep -cE '^[l]' "$OUT_MANIFEST" || true)
echo "Symlinks in tar listing: $SYMS_IN_TAR"

echo "Done."
echo "Preview:"
tar -tzf "$OUT_TAR" | head
