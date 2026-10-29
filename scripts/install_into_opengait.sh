#!/usr/bin/env bash
# Enlaza el modelo y la pérdida de este repo dentro de OpenGait (registro automático de clases).
# Usa enlaces simbólicos: editar en el repo se refleja en OpenGait sin volver a copiar.
set -euo pipefail

DISTILL_ROOT="${DISTILL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OPENGAIT_ROOT="${OPENGAIT_ROOT:-/opt3/data/julieta/OpenGait}"

link() {
  local src="$1" dst="$2"
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then
    mv "$dst" "$dst.bak.$(date +%Y%m%d%H%M%S)"
    echo "Copia previa respaldada: $dst.bak.*"
  fi
  ln -sfn "$src" "$dst"
  echo "$dst -> $src"
}

link "$DISTILL_ROOT/models/student_distill.py"  "$OPENGAIT_ROOT/opengait/modeling/models/student_distill.py"
link "$DISTILL_ROOT/losses/distillation_loss.py" "$OPENGAIT_ROOT/opengait/modeling/losses/distillation_loss.py"
