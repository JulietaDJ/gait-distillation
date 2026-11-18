#!/usr/bin/env python3
"""
Empaqueta las siluetas PNG generadas con YOLO en archivos .pkl,
uno por secuencia (sujeto/condición/ángulo), sin recortar ni redimensionar.

Estructura de entrada:
    CASIA-B-yolo/silhouettes/001/bg-01/000/001.png, 002.png, ...
Estructura de salida:
    CASIA-B-yolo-pkl/001/bg-01/000/000-sils.pkl   (array [T, 64, 44])
"""

import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import cv2


def empaquetar_secuencia(carpeta_frames, archivo_salida):
    """
    Lee todos los PNGs de una carpeta, los apila y guarda un .pkl.
    Devuelve el número de frames empaquetados.
    """
    # Lista de frames ordenados por nombre (001.png, 002.png, ...)
    frames = sorted(carpeta_frames.glob("*.png"))
    if not frames:
        return 0

    # Cargar cada frame como escala de grises (silueta binaria)
    imgs = []
    for f in frames:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] No se pudo leer: {f}")
            continue
        imgs.append(img)

    if not imgs:
        return 0

    # Apilar en un array [T, H, W]
    secuencia = np.stack(imgs, axis=0).astype(np.uint8)

    # Guardar como .pkl
    with open(archivo_salida, "wb") as fh:
        pickle.dump(secuencia, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return secuencia.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/opt3/data/julieta/CASIA-B-yolo/silhouettes",
                        help="Carpeta raíz de siluetas (con subcarpetas sujeto/cond/ángulo)")
    parser.add_argument("--output", default="/opt3/data/julieta/CASIA-B-yolo-pkl",
                        help="Carpeta destino para los .pkl")
    parser.add_argument("--max-sujetos", type=int, default=0,
                        help="Procesar solo N sujetos (0=todos)")
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    sujetos = sorted([d for d in in_root.iterdir() if d.is_dir()])
    if args.max_sujetos > 0:
        sujetos = sujetos[:args.max_sujetos]

    total_secuencias = 0
    total_frames = 0

    for sujeto in sujetos:
        for cond in sorted(sujeto.iterdir()):
            if not cond.is_dir():
                continue
            for ang in sorted(cond.iterdir()):
                if not ang.is_dir():
                    continue

                # Crear carpeta destino equivalente
                dst_dir = out_root / sujeto.name / cond.name / ang.name
                dst_dir.mkdir(parents=True, exist_ok=True)

                # Nombre del archivo .pkl: <ángulo>-sils.pkl
                dst_file = dst_dir / f"{ang.name}-sils.pkl"

                n_frames = empaquetar_secuencia(ang, dst_file)
                if n_frames > 0:
                    total_secuencias += 1
                    total_frames += n_frames
                    print(f"  {sujeto.name}/{cond.name}/{ang.name}: {n_frames} frames → {dst_file.name}")

    print(f"\n=== Resumen ===")
    print(f"Secuencias empaquetadas: {total_secuencias}")
    print(f"Frames totales:          {total_frames}")
    print(f"Salida:                  {out_root}")


if __name__ == "__main__":
    main()
