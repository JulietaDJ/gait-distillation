#!/usr/bin/env python3
"""
Fase 2: A/B de reconocimiento de marcha con y sin el backbone destilado.

Pasos, todos reanudables (lo ya evaluado no se repite salvo --reevaluar):
  1. Verifica que las dos configuraciones solo difieren en restore_hint/save_name:
     cualquier otra diferencia invalidaría la comparación.
  2. Puntos de partida, sin entrenar reconocimiento: Rank-1/mAP con el embedding del
     backbone (HPP, sin cabeza) para un backbone aleatorio y para el destilado. Dicen
     cuánta identidad trae ya la representación de la fase 1.
  3. Entrena cada brazo con OpenGait. En el destilado exige que el log registre la
     carga del checkpoint: si no, entrenaría desde cero en silencio.
  4. Evalúa cada checkpoint guardado (cada save_iter) con evaluar_reconocimiento.py:
     Rank-1, Rank-5 y mAP por condición y por ángulo. Con varios checkpoints se ve
     también si el destilado converge antes, no solo dónde termina.
  5. comparar_fase2.py: tabla y figuras.

No usa la fase test de OpenGait: su evaluador exige batch_size == nº de GPUs y solo
reporta Rank-1. El nuestro reproduce su Rank-1 exacto y añade Rank-5, mAP y ángulos.

Uso:
    python scripts/run_fase2.py --opengait-root ~/OpenGait --gpu 0 \
        --backbone /ruta/backbone_fase1_20k_deepgaitv2.pt
    python scripts/run_fase2.py --skip-train          # solo evaluar lo ya entrenado
    python scripts/run_fase2.py --only distill        # un solo brazo
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RE_RESTORE = re.compile(r"Restore Parameters from")
RE_ITER = re.compile(r"-(\d+)\.pt$")


def comparar_configs(a, b):
    """Aborta si las configuraciones difieren en algo que no sea restore_hint/save_name."""
    ca, cb = yaml.safe_load(Path(a).read_text()), yaml.safe_load(Path(b).read_text())
    permitido = {"restore_hint", "save_name"}
    problemas = []
    for sec in set(ca) | set(cb):
        va, vb = ca.get(sec), cb.get(sec)
        if va == vb:
            continue
        if not (isinstance(va, dict) and isinstance(vb, dict)):
            problemas.append(sec)
            continue
        for k in set(va) | set(vb):
            if va.get(k) != vb.get(k) and k not in permitido:
                problemas.append(f"{sec}.{k}: {va.get(k)!r} vs {vb.get(k)!r}")
    if problemas:
        sys.exit("Las configuraciones difieren en algo más que restore_hint/save_name:\n  "
                 + "\n  ".join(problemas))
    return ca, cb


def ejecutar(cmd, cwd, gpu, log):
    """Lanza un proceso, replica su salida en pantalla y en log, devuelve el texto."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    print(f"\n>>> {' '.join(map(str, cmd))}\n    log: {log}", flush=True)
    texto = []
    with open(log, "w") as fh, subprocess.Popen(
            list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for linea in p.stdout:
            sys.stdout.write(linea)
            fh.write(linea)
            texto.append(linea)
        rc = p.wait()
    if rc != 0:
        sys.exit(f"Terminó con código {rc}; revisa {log}")
    return "".join(texto)


def entrenar(cfg, og, gpu, log_dir):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
           "opengait/main.py", "--cfgs", Path(cfg).resolve(), "--phase", "train", "--log_to_file"]
    return ejecutar(cmd, og, gpu, log_dir / f"{Path(cfg).stem}_train.log")


def checkpoints(og, cfg):
    c = yaml.safe_load(Path(cfg).read_text())
    nombre = c["trainer_cfg"]["save_name"]
    d = og / "output" / c["data_cfg"]["dataset_name"] / c["model_cfg"]["model"] / nombre / "checkpoints"
    cks = [(int(RE_ITER.search(p.name).group(1)), p) for p in d.glob(f"{nombre}-*.pt") if RE_ITER.search(p.name)]
    return sorted(cks), d


def evaluar(etiqueta, ck, cfg, embedding, destino, args, og, log_dir):
    if destino.exists() and not args.reevaluar:
        print(f"[ya evaluado] {destino}")
        return
    cmd = [sys.executable, REPO / "data/evaluar_reconocimiento.py", "--checkpoint", ck, "--cfg", Path(cfg).resolve(),
           "--opengait-root", og, "--embedding", embedding, "--salida", destino, "--etiqueta", etiqueta]
    if args.fp16:
        cmd.append("--fp16")
    ejecutar(cmd, REPO, args.gpu, log_dir / f"eval_{destino.stem}.log")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opengait-root", default="~/OpenGait")
    p.add_argument("--scratch", default=str(REPO / "configs/fase2_scratch.yaml"))
    p.add_argument("--distill", default=str(REPO / "configs/fase2_distill.yaml"))
    p.add_argument("--backbone", default="", help="Backbone *_deepgaitv2.pt de la fase 1 (sustituye restore_hint)")
    p.add_argument("--gpu", default="0")
    p.add_argument("--skip-train", action="store_true", help="Solo evaluar checkpoints existentes")
    p.add_argument("--only", choices=["scratch", "distill"], help="Ejecutar un solo brazo")
    p.add_argument("--solo-final", action="store_true", help="Evaluar solo el último checkpoint de cada brazo")
    p.add_argument("--sin-cero", action="store_true", help="Omitir los puntos de partida sin entrenar")
    p.add_argument("--reevaluar", action="store_true", help="Repetir evaluaciones ya guardadas")
    p.add_argument("--fp16", action="store_true", help="Evaluar con autocast, como enable_float16 de OpenGait")
    p.add_argument("--salida", default="resultados_fase2")
    args = p.parse_args()

    og = Path(args.opengait_root).expanduser().resolve()
    if not (og / "opengait/main.py").exists():
        sys.exit(f"No encuentro opengait/main.py en {og}")
    salida = Path(args.salida).resolve()
    log_dir, met_dir = salida / "logs", salida / "metricas"
    log_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    cfg_s, cfg_d = Path(args.scratch), Path(args.distill)
    if args.backbone:                           # copia resuelta: el yaml del repo no se toca
        c = yaml.safe_load(cfg_d.read_text())
        c["trainer_cfg"]["restore_hint"] = str(Path(args.backbone).expanduser().resolve())
        cfg_d = salida / "fase2_distill.resuelto.yaml"
        cfg_d.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
    _, cb = comparar_configs(cfg_s, cfg_d)
    hint = cb["trainer_cfg"]["restore_hint"]
    if not isinstance(hint, str) or not Path(hint).exists():
        sys.exit(f"El backbone destilado no existe: {hint!r} (pásalo con --backbone)")
    print(f"[ok] configuraciones alineadas | backbone destilado: {hint}")

    t0 = datetime.now()
    if not args.sin_cero:
        evaluar("cero_aleatorio", "aleatorio", cfg_s, "backbone", met_dir / "cero_aleatorio-00000.json", args, og, log_dir)
        evaluar("cero_destilado", hint, cfg_d, "backbone", met_dir / "cero_destilado-00000.json", args, og, log_dir)

    brazos = [("scratch", cfg_s), ("distill", cfg_d)]
    if args.only:
        brazos = [b for b in brazos if b[0] == args.only]
    for nombre, cfg in brazos:
        if not args.skip_train:
            texto = entrenar(cfg, og, args.gpu, log_dir)
            if nombre == "distill" and not RE_RESTORE.search(texto):
                sys.exit("El brazo destilado NO cargó el checkpoint (falta 'Restore Parameters from'). "
                         "El experimento no sería válido; revisa restore_hint.")
        cks, d = checkpoints(og, cfg)
        if not cks:
            sys.exit(f"No hay checkpoints de {nombre} en {d}")
        if args.solo_final:
            cks = cks[-1:]
        for it, ck in cks:
            evaluar(nombre, ck, cfg, "fcs", met_dir / f"{nombre}-{it:05d}.json", args, og, log_dir)

    subprocess.run([sys.executable, REPO / "scripts/comparar_fase2.py",
                    "--metricas", met_dir, "--salida", salida], check=True)
    print(f"\nTiempo total: {datetime.now() - t0} | resultados en {salida}/")


if __name__ == "__main__":
    main()
