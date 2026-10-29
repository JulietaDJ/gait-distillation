"""
Lanzador de la destilación sobre OpenGait, sin monkeypatching.

StudentDistill define su propio get_loader e inputs_pretreament, así que aquí
solo se preparan las rutas y se ejecuta opengait/main.py tal cual.

Requisito previo (una vez): scripts/install_into_opengait.sh
Uso:
    python -m torch.distributed.run --nproc_per_node=1 scripts/train_distill.py \
        --cfgs <repo>/configs/student_distill_casiab.yaml --phase train --log_to_file

Rutas configurables con OPENGAIT_ROOT y DISTILL_ROOT.
"""
import os
import runpy
import sys
from pathlib import Path

DISTILL_ROOT = Path(os.environ.get("DISTILL_ROOT", Path(__file__).resolve().parents[1]))
OPENGAIT_ROOT = Path(os.environ.get("OPENGAIT_ROOT", "/opt3/data/julieta/OpenGait"))

# Resolver --cfgs a ruta absoluta antes del chdir (OpenGait trabaja relativo a su raíz).
if "--cfgs" in sys.argv:
    i = sys.argv.index("--cfgs") + 1
    sys.argv[i] = str(Path(sys.argv[i]).resolve())

# Solo la carpeta data/ del repo (no la raíz): evita colisión con el paquete 'data' de OpenGait.
sys.path.insert(0, str(DISTILL_ROOT / "data"))
sys.path.insert(0, str(OPENGAIT_ROOT / "opengait"))
os.chdir(OPENGAIT_ROOT)

for rel in ("opengait/modeling/models/student_distill.py",
            "opengait/modeling/losses/distillation_loss.py"):
    if not (OPENGAIT_ROOT / rel).exists():
        sys.exit(f"Falta {rel} en OpenGait. Ejecuta scripts/install_into_opengait.sh")

if __name__ == "__main__":
    sys.argv[0] = str(OPENGAIT_ROOT / "opengait" / "main.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
