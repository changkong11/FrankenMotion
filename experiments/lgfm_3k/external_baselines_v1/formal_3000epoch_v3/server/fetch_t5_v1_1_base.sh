#!/usr/bin/env bash
# Download the exact T5-v1.1-base files required by MoScale from ModelScope,
# then verify every SHA-256 before the training queue can use them.
set -Eeuo pipefail
DEST=${1:-/data_disk/wulw/baselines/weights/t5-v1_1-base}
BASE_URL='https://www.modelscope.cn/api/v1/models/google/t5-v1_1-base/repo?Revision=master&FilePath='
mkdir -p "$DEST"
declare -A EXPECTED=(
  [config.json]=ca0b58012f673465b94eb884d801148e44c86346380ba03cc24c268b20ecc505
  [pytorch_model.bin]=1d51351caeff6b750968985458a25b74622657f492c01cc62dbf3a61074725eb
  [spiece.model]=d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86
  [special_tokens_map.json]=4720c0fddbe4c5991334f85ad7073d9bd0a294a8ba4641a2f8dab614ca825949
  [tokenizer_config.json]=b971dce1d2805c2a66da8657156e7114a30501c6ba602fc947c8bf607a3ead2d
)
for name in config.json pytorch_model.bin spiece.model special_tokens_map.json tokenizer_config.json; do
  path="$DEST/$name"
  if [ -f "$path" ] && [ "$(sha256sum "$path" | awk '{print $1}')" = "${EXPECTED[$name]}" ]; then
    printf 'verified_existing %s\n' "$name"
    continue
  fi
  rm -f "$path.part"
  printf 'download %s\n' "$name"
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - --output "$path.part" "${BASE_URL}${name}"
  actual=$(sha256sum "$path.part" | awk '{print $1}')
  test "$actual" = "${EXPECTED[$name]}"
  mv "$path.part" "$path"
  printf 'verified_download %s\n' "$name"
done
printf 't5_ready path=%s\n' "$DEST"