#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/disassembly/raw"
ANNOTATED_DIR="$SCRIPT_DIR/disassembly/annotated"

case "$RAW_DIR:$ANNOTATED_DIR" in
    "$SCRIPT_DIR/disassembly/raw:$SCRIPT_DIR/disassembly/annotated") ;;
    *) echo "Refusing to replace unexpected disassembly paths" >&2; exit 1 ;;
esac
rm -rf -- "$RAW_DIR" "$ANNOTATED_DIR"

python3 "$SCRIPT_DIR/disassemble.py" --output "$RAW_DIR"
mkdir -p "$ANNOTATED_DIR"
for raw in "$RAW_DIR"/*.asm.gz; do
    name="$(basename "$raw" .asm.gz)"
    python3 "$SCRIPT_DIR/annotate_disassembly.py" \
        --elf "$SCRIPT_DIR/artifacts/measured/pa_scheduler_kernel.o" \
        --raw "$raw" \
        --source-root "$SCRIPT_DIR" \
        --output "$ANNOTATED_DIR/${name}.source.asm.gz"
done
mkdir -p "$SCRIPT_DIR/disassembly/key_flow"
python3 "$SCRIPT_DIR/annotate_disassembly.py" \
    --elf "$SCRIPT_DIR/artifacts/measured/pa_scheduler_kernel.o" \
    --raw "$RAW_DIR/03_pa_scheduler_lazy_sample_callback_orchestration_aic_9bd281ff.asm.gz" \
    --source-root "$SCRIPT_DIR" \
    --output "$SCRIPT_DIR/disassembly/key_flow/aic_compete_first_lazy.source.asm" \
    --anchor common/pa_scheduler_core.h:1382 --before 558 --after 162
python3 "$SCRIPT_DIR/annotate_disassembly.py" \
    --elf "$SCRIPT_DIR/artifacts/measured/pa_scheduler_kernel.o" \
    --raw "$RAW_DIR/03_pa_scheduler_lazy_sample_callback_orchestration_aic_9bd281ff.asm.gz" \
    --source-root "$SCRIPT_DIR" \
    --output "$SCRIPT_DIR/disassembly/key_flow/aic_lazy_input_policy.source.asm" \
    --anchor common/pa_frontend.h:703 --before 160 --after 640
(
    cd "$SCRIPT_DIR/disassembly"
    find raw annotated key_flow -type f -print0 | sort -z | xargs -0 sha256sum
) > "$SCRIPT_DIR/disassembly/published.sha256"
