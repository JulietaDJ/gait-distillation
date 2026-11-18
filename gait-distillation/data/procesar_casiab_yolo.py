#!/usr/bin/env python3
"""
Procesa todo CASIA-B: detecta personas con YOLO, genera siluetas binarias
y RGB recortados, ambos normalizados a 64x44 (alto x ancho).
Conserva la estructura de carpetas original.
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


# --- Configuración ---
OUTPUT_SIZE = (64, 44)  # (alto, ancho)
CONF_THRESHOLD = 0.3
PERSON_CLASS_ID = 0


def normalizar_con_padding(img, output_size=OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST):
    """
    Normaliza una imagen (silueta o RGB) al tamaño output_size (alto, ancho)
    manteniendo la proporción y rellenando con negro.
    Siluetas: INTER_NEAREST (mantiene la máscara binaria).
    RGB: INTER_AREA al reducir (evita aliasing), INTER_LINEAR al ampliar.
    """
    h, w = img.shape[:2]
    out_h, out_w = output_size

    # Escalar manteniendo proporción
    escala = min(out_h / h, out_w / w)
    nuevo_h = max(1, int(h * escala))
    nuevo_w = max(1, int(w * escala))

    img_resized = cv2.resize(img, (nuevo_w, nuevo_h),
                             interpolation=interpolation)

    # Crear lienzo negro
    if len(img.shape) == 2:
        lienzo = np.zeros((out_h, out_w), dtype=np.uint8)
    else:
        lienzo = np.zeros((out_h, out_w, img.shape[2]), dtype=np.uint8)

    # Centrar
    y_off = (out_h - nuevo_h) // 2
    x_off = (out_w - nuevo_w) // 2
    lienzo[y_off:y_off+nuevo_h, x_off:x_off+nuevo_w] = img_resized

    return lienzo


def pad_cuadrado(img, pad_factor):
    """Centra la imagen en un cuadrado negro de pad_factor x lado mayor
    (réplica de pad_cropped_image() del demo de VHuman3D)."""
    h, w = img.shape[:2]
    side = int(max(h, w) * pad_factor)
    lienzo = np.zeros((side, side) + img.shape[2:], dtype=img.dtype)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    lienzo[y_off:y_off + h, x_off:x_off + w] = img
    return lienzo


def extraer_persona_y_silueta(frame, modelo_yolo):
    """
    Detecta la persona principal en el frame y devuelve:
    - roi_rgb: recorte RGB de la persona
    - roi_mask: silueta binaria dentro del mismo recorte
    Devuelve (None, None) si no se detecta persona.
    """
    results = modelo_yolo(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    personas = []
    for box in results.boxes:
        if int(box.cls[0]) == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            personas.append((area, x1, y1, x2, y2))

    if not personas:
        return None, None

    # La persona de mayor área
    personas.sort(reverse=True)
    _, x1, y1, x2, y2 = personas[0]

    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi_rgb = frame[y1:y2, x1:x2]

    # Generar máscara binaria
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Limpiar con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Contorno más grande
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)

    # Descartar contornos demasiado pequeños (probablemente ruido)
    if cv2.contourArea(c) < 50:
        return None, None

    x_c, y_c, w_c, h_c = cv2.boundingRect(c)

    # Recortar a la bounding box del contorno (silueta ajustada)
    sil_crop = mask[y_c:y_c+h_c, x_c:x_c+w_c]
    rgb_crop = roi_rgb[y_c:y_c+h_c, x_c:x_c+w_c]

    return rgb_crop, sil_crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/CASIA-B/image",
                        help="Carpeta raíz de RGB")
    parser.add_argument("--output", default="/opt3/data/julieta/CASIA-B-yolo",
                        help="Carpeta destino")
    parser.add_argument("--max-sujetos", type=int, default=0,
                        help="Procesar solo N sujetos (0=todos, útil para pruebas)")
    parser.add_argument("--rgb-size", type=int, nargs=2, metavar=("H", "W"),
                        default=list(OUTPUT_SIZE),
                        help="Tamaño del RGB para el maestro (por defecto igual a la silueta). "
                             "Recomendado para VHuman3D: 256 256 con --rgb-pad-factor 1.2")
    parser.add_argument("--rgb-pad-factor", type=float, default=0.0,
                        help="Si > 0, el RGB se rellena a un cuadrado de factor x lado mayor "
                             "(como el demo de VHuman3D) antes de redimensionar; requiere "
                             "--rgb-size cuadrado. 0 = comportamiento original.")
    args = parser.parse_args()
    rgb_size = tuple(args.rgb_size)
    if args.rgb_pad_factor > 0 and rgb_size[0] != rgb_size[1]:
        parser.error("--rgb-pad-factor requiere un --rgb-size cuadrado")

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_sil = out_root / "silhouettes"
    out_rgb = out_root / "rgb"
    out_sil.mkdir(parents=True, exist_ok=True)
    out_rgb.mkdir(parents=True, exist_ok=True)

    # Cargar YOLO
    print("Cargando YOLO...")
    modelo = YOLO("yolov8n.pt")

    sujetos = sorted([d for d in in_root.iterdir() if d.is_dir()])
    if args.max_sujetos > 0:
        sujetos = sujetos[:args.max_sujetos]

    total_frames = 0
    total_ok = 0
    total_fallos = 0

    for sujeto in tqdm(sujetos, desc="Sujetos"):
        for cond in sorted(sujeto.iterdir()):
            if not cond.is_dir():
                continue
            for ang in sorted(cond.iterdir()):
                if not ang.is_dir():
                    continue

                frames = sorted(ang.glob("*.png"))
                if not frames:
                    continue

                # Crear carpetas destino equivalentes
                dst_sil = out_sil / sujeto.name / cond.name / ang.name
                dst_rgb = out_rgb / sujeto.name / cond.name / ang.name
                dst_sil.mkdir(parents=True, exist_ok=True)
                dst_rgb.mkdir(parents=True, exist_ok=True)

                for frame_path in frames:
                    total_frames += 1
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        total_fallos += 1
                        continue

                    rgb_crop, sil_crop = extraer_persona_y_silueta(frame, modelo)

                    if rgb_crop is None or sil_crop is None:
                        total_fallos += 1
                        continue

                    # Normalizar ambos al mismo tamaño
                    reduce = rgb_crop.shape[0] > rgb_size[0]
                    interp = cv2.INTER_AREA if reduce else cv2.INTER_LINEAR
                    if args.rgb_pad_factor > 0:
                        rgb_sq = pad_cuadrado(rgb_crop, args.rgb_pad_factor)
                        rgb_norm = cv2.resize(rgb_sq, (rgb_size[1], rgb_size[0]),
                                              interpolation=interp)
                    else:
                        rgb_norm = normalizar_con_padding(rgb_crop, rgb_size, interp)
                    sil_norm = normalizar_con_padding(sil_crop, OUTPUT_SIZE)

                    # Guardar con el mismo nombre base
                    nombre_base = frame_path.stem  # ej. "030"
                    cv2.imwrite(str(dst_sil / f"{nombre_base}.png"), sil_norm)
                    cv2.imwrite(str(dst_rgb / f"{nombre_base}.png"), rgb_norm)
                    total_ok += 1

    print(f"\n=== Resumen ===")
    print(f"Frames procesados: {total_frames}")
    print(f"Generados OK:      {total_ok}")
    print(f"Fallos:            {total_fallos}")
    print(f"Salida siluetas:   {out_sil}")
    print(f"Salida RGB:        {out_rgb}")


if __name__ == "__main__":
    main()
