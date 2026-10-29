#!/usr/bin/env python3
"""
Media y desviación por canal de las features del maestro, SOLO sobre TRAIN_SET.

Uso:
    python calcular_stats_maestro.py \
        --teacher-root /opt3/data/julieta/vhuman_features_v2 \
        --partition /opt3/data/julieta/OpenGait/datasets/CASIA-B/CASIA-B.json \
        --output /opt3/data/julieta/vhuman_features_v2/teacher_stats.npz

El .npz resultante (mean, std con forma [768], o [16, 768] con --per-part)
se indica en data_cfg.teacher_stats y el collate estandariza el maestro.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-root", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-part", action="store_true", help="Estadísticas por (parte, canal)")
    p.add_argument("--max-frames-seq", type=int, default=0, help="Submuestrear frames por secuencia (0 = todos)")
    args = p.parse_args()

    train = json.loads(Path(args.partition).read_text())["TRAIN_SET"]
    root = Path(args.teacher_root)
    axes = (0,) if args.per_part else (0, 1)

    s = ss = None
    count = 0
    n_seq = 0
    for pid in train:
        for f in sorted((root / pid).glob("*/*/features.npy")):
            x = np.load(f, mmap_mode="r")
            if args.max_frames_seq and len(x) > args.max_frames_seq:
                x = x[np.linspace(0, len(x) - 1, args.max_frames_seq).astype(int)]
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 4:
                raise SystemExit(f"{f} está en formato antiguo [T,16,16,768]; reextrae primero")
            if s is None:
                s = np.zeros(x.shape[1:] if args.per_part else x.shape[-1:])
                ss = np.zeros_like(s)
            s += x.sum(axis=axes)
            ss += (x ** 2).sum(axis=axes)
            count += x.shape[0] * (1 if args.per_part else x.shape[1])
            n_seq += 1

    if count == 0:
        raise SystemExit("No se encontraron features para TRAIN_SET")
    mean = s / count
    std = np.sqrt(np.maximum(ss / count - mean ** 2, 0.0))
    np.savez(args.output, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Secuencias: {n_seq} | muestras: {count} | mean∈[{mean.min():.3f}, {mean.max():.3f}] "
          f"| std∈[{std.min():.3f}, {std.max():.3f}] -> {args.output}")


if __name__ == "__main__":
    main()
