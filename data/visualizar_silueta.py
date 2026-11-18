#!/usr/bin/env python3
"""
Pinta la calidad de la destilación sobre la silueta real, franja por franja.

El estudiante reduce cada frame a 16 franjas horizontales: el backbone baja la
silueta de 64x44 a un mapa de features de 16x11 (stride vertical 4) y la HPP
toma una fila del mapa por franja (view(n, c, 16, -1)). La franja p cubre, por
tanto, exactamente las filas [4p, 4p+4) de la silueta de entrada. Eso permite
pintar la métrica de cada franja encima del cuerpo, en vez de en una gráfica
desligada de la imagen.

Por cada secuencia genera:
    <salida>/<sujeto>-<cond>-<angulo>_tira.png   tira estática de frames
    <salida>/<sujeto>-<cond>-<angulo>.gif        animación con curva temporal

En las imágenes el color codifica la métrica (viridis, claro = mejor). Con
--metric l1 la escala se invierte, de modo que "claro y barra larga = mejor"
en los dos casos.

Los textos dibujados van sin acentos a propósito: cv2.putText solo rasteriza
ASCII y los acentos saldrían como cajas.

Uso:
    python data/visualizar_silueta.py \
        --checkpoint ~/OpenGait/output/CASIA-B/StudentDistill/StudentDistill_Phase1/checkpoints/StudentDistill_Phase1-10000.pt \
        --cfg configs/student_distill_casiab.yaml \
        --opengait-root ~/OpenGait \
        --salida visualizaciones
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from evaluar_destilacion import construir_modelo, cargar_pesos

PARTS = 16
FONDO = 16                      # gris del lienzo
CUERPO = 232                    # gris del cuerpo antes de teñir
LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                        cv2.COLORMAP_VIRIDIS).reshape(256, 3)


# --------------------------------------------------------------------------- #
#  Primitivas de dibujo
# --------------------------------------------------------------------------- #
def color(frac):
    """frac en [0, 1] -> BGR del colormap."""
    return LUT[int(np.clip(frac, 0.0, 1.0) * 255)]


def bgr(c):
    return tuple(int(v) for v in c)


def texto(img, s, org, escala=0.42, col=(210, 210, 210), grosor=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, escala, col, grosor, cv2.LINE_AA)


def mezclar(region, c, alpha):
    return (region * (1.0 - alpha) + np.asarray(c, np.float32) * alpha).astype(np.uint8)


def etiquetar(panel, s, alto=22):
    """Añade una banda inferior con el texto centrado."""
    banda = np.full((alto, panel.shape[1], 3), FONDO, np.uint8)
    if s:
        (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        texto(banda, s, (max(2, (panel.shape[1] - w) // 2), alto - 7))
    return np.vstack([panel, banda])


def fila(paneles, hueco=10):
    """Concatena paneles en horizontal, rellenando hasta la altura máxima."""
    alto = max(p.shape[0] for p in paneles)
    piezas = []
    for p in paneles:
        if p.shape[0] < alto:
            p = np.vstack([p, np.full((alto - p.shape[0], p.shape[1], 3), FONDO, np.uint8)])
        piezas += [p, np.full((alto, hueco, 3), FONDO, np.uint8)]
    return np.hstack(piezas[:-1])


def con_titulo(img, titulo, sub=""):
    cab = np.full((50 if sub else 34, img.shape[1], 3), FONDO, np.uint8)
    texto(cab, titulo, (10, 21), 0.52, (240, 240, 240))
    if sub:
        texto(cab, sub, (10, 42), 0.40, (150, 150, 150))
    return np.vstack([cab, img, np.full((8, img.shape[1], 3), FONDO, np.uint8)])


class Escala:
    """Valores de la métrica -> fracción 0..1 del colormap (1 = mejor)."""

    def __init__(self, vmin, vmax, invertir):
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.invertir = bool(invertir)

    def frac(self, v):
        f = (np.asarray(v, np.float64) - self.vmin) / max(1e-12, self.vmax - self.vmin)
        f = np.clip(f, 0.0, 1.0)
        return 1.0 - f if self.invertir else f


# --------------------------------------------------------------------------- #
#  Paneles
# --------------------------------------------------------------------------- #
def render_silueta(sil, vals, esc, escala_px, tenir=True):
    """Silueta ampliada, con las 16 franjas teñidas por su valor de la métrica."""
    h, w = sil.shape
    up = cv2.resize(sil, (w * escala_px, h * escala_px), interpolation=cv2.INTER_NEAREST)
    cuerpo = up > 127
    img = np.full((*up.shape, 3), FONDO, np.uint8)
    img[cuerpo] = CUERPO
    if not tenir:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (55, 55, 55), 1)
        return img

    paso = h // PARTS * escala_px
    for p in range(PARTS):
        c = color(esc.frac(vals[p]))
        reg = img[p * paso:(p + 1) * paso]
        m = cuerpo[p * paso:(p + 1) * paso]
        reg[m] = mezclar(reg[m], c, 0.85)        # el cuerpo lleva el color saturado
        reg[~m] = mezclar(reg[~m], c, 0.20)      # el fondo, un tinte que marca la banda
    for p in range(1, PARTS):
        cv2.line(img, (0, p * paso), (img.shape[1] - 1, p * paso), (60, 60, 60), 1)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (55, 55, 55), 1)
    return img


def render_barras(vals, esc, alto, ancho, ref=None):
    """Barras horizontales alineadas fila a fila con las franjas de la silueta."""
    img = np.full((alto, ancho, 3), FONDO, np.uint8)
    x0, x1 = 26, ancho - 46
    paso = alto // PARTS

    if ref is not None:
        xr = x0 + int((x1 - x0) * esc.frac(ref))
        for y in range(0, alto, 6):
            cv2.line(img, (xr, y), (xr, min(alto - 1, y + 3)), (115, 115, 115), 1)

    for p in range(PARTS):
        y0 = p * paso
        f = float(esc.frac(vals[p]))
        cv2.rectangle(img, (x0, y0 + 3), (x0 + max(1, int((x1 - x0) * f)), y0 + paso - 3),
                      bgr(color(f)), -1)
        texto(img, f"{p:2d}", (2, y0 + paso - 7), 0.36, (130, 130, 130))
        texto(img, f"{vals[p]:.3f}", (x1 + 4, y0 + paso - 7), 0.36, (205, 205, 205))
    return img


def render_curva(serie, cur, esc, alto, ancho, ref=None):
    """Media por frame de la métrica, con cursor en el frame actual."""
    img = np.full((alto, ancho, 3), FONDO, np.uint8)
    x0, x1, y0, y1 = 44, ancho - 12, 16, alto - 26
    cv2.rectangle(img, (x0, y0), (x1, y1), (70, 70, 70), 1)

    def yv(v):
        f = np.clip((v - esc.vmin) / max(1e-12, esc.vmax - esc.vmin), 0.0, 1.0)
        return int(y1 - f * (y1 - y0))

    if ref is not None:
        for x in range(x0, x1, 6):
            cv2.line(img, (x, yv(ref)), (min(x1, x + 3), yv(ref)), (115, 115, 115), 1)
        texto(img, "trivial", (x0 + 4, max(y0 + 10, yv(ref) - 4)), 0.34, (125, 125, 125))

    n = len(serie)
    xs = [x0 + int((x1 - x0) * i / max(1, n - 1)) for i in range(n)]
    cv2.polylines(img, [np.array([[xs[i], yv(serie[i])] for i in range(n)], np.int32)],
                  False, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.line(img, (xs[cur], y0), (xs[cur], y1), (235, 170, 90), 1)
    cv2.circle(img, (xs[cur], yv(serie[cur])), 3, bgr(color(esc.frac(serie[cur]))), -1)

    texto(img, f"{esc.vmax:.2f}", (4, y0 + 10), 0.36, (140, 140, 140))
    texto(img, f"{esc.vmin:.2f}", (4, y1), 0.36, (140, 140, 140))
    texto(img, f"frame {cur + 1}/{n}", (x0 + 4, alto - 8), 0.38, (180, 180, 180))
    return img


def render_cbar(esc, alto, ancho=18):
    """Barra de color vertical, siempre con el mejor valor arriba."""
    img = np.full((alto, ancho + 54, 3), FONDO, np.uint8)
    arr, aba = (esc.vmin, esc.vmax) if esc.invertir else (esc.vmax, esc.vmin)
    for i in range(alto):
        img[i, :ancho] = color(esc.frac(arr + (aba - arr) * i / max(1, alto - 1)))
    cv2.rectangle(img, (0, 0), (ancho - 1, alto - 1), (70, 70, 70), 1)
    for k in range(5):
        v = arr + (aba - arr) * k / 4
        y = int(np.clip(k / 4 * (alto - 1), 9, alto - 4))
        texto(img, f"{v:.2f}", (ancho + 5, y + 4), 0.36, (195, 195, 195))
    return img


# --------------------------------------------------------------------------- #
#  Inferencia
# --------------------------------------------------------------------------- #
@torch.no_grad()
def metricas(net, trf, sils, teacher, t_mean, t_std, device, lote=64):
    """Devuelve cos y Smooth L1 por [frame, franja]."""
    cos, l1, aviso = [], [], True
    for i in range(0, len(sils), lote):
        x = np.stack([trf(s) for s in sils[i:i + lote]]).astype(np.float32)
        x = torch.from_numpy(x).to(device)[:, None]                    # [n, 1, H, W]
        f = net.layer4(net.layer3(net.layer2(net.layer1(net.layer0(x)))))
        if aviso:
            aviso = False
            if f.shape[-2] != PARTS:
                print(f"  [warn] mapa de features {tuple(f.shape[-2:])}: la franja ya no "
                      f"corresponde a {sils.shape[1] // PARTS} filas exactas de la silueta")
        s = net.proj_head(net.HPP(f).permute(0, 2, 1)).double()         # [n, 16, 768]

        t = torch.from_numpy(np.asarray(teacher[i:i + lote], dtype=np.float32)).to(device)
        if t_mean is not None:
            t = (t - t_mean) / t_std
        t = t.double()

        cos.append(F.cosine_similarity(s, t, dim=-1).cpu().numpy())
        l1.append(F.smooth_l1_loss(s, t, beta=1.0, reduction='none').mean(-1).cpu().numpy())
    return np.concatenate(cos, 0), np.concatenate(l1, 0)


def elegir(sil_root, teacher_root, pids, pedidas, conds, angulo):
    """Secuencias explícitas, o la primera identidad disponible por condición."""
    if pedidas:
        return [Path(s.strip("/")) for s in pedidas]
    out = []
    for c in conds:
        for pid in pids:
            rel = Path(pid) / c / angulo
            if (teacher_root / rel / "features.npy").exists() and any((sil_root / rel).glob("*.pkl")):
                out.append(rel)
                break
    return out


def guardar_gif(frames, destino, fps, colores):
    try:
        from PIL import Image
    except ImportError:
        carpeta = destino.with_suffix("")
        carpeta.mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(frames):
            cv2.imwrite(str(carpeta / f"{k:04d}.png"), f)
        print(f"  [gif] Pillow no disponible: frames sueltos en {carpeta}")
        return
    ims = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
           .convert("P", palette=Image.ADAPTIVE, colors=colores) for f in frames]
    ims[0].save(destino, save_all=True, append_images=ims[1:],
                duration=max(20, int(round(1000 / fps))), loop=0, optimize=True, disposal=2)
    print(f"  [gif]  {destino}  ({len(frames)} frames, {destino.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cfg", default="configs/student_distill_casiab.yaml")
    p.add_argument("--opengait-root", default="~/OpenGait")
    p.add_argument("--partition", default="", help="Por defecto, el del yaml resuelto contra --opengait-root")
    p.add_argument("--split", default="TEST_SET", choices=["TEST_SET", "TRAIN_SET"])
    p.add_argument("--seqs", nargs="*", default=[],
                   help="Rutas tipo 075/nm-05/090 (vacío = una por condición)")
    p.add_argument("--conds", nargs="*", default=["nm-05", "bg-01", "cl-01"])
    p.add_argument("--angulo", default="090")
    p.add_argument("--metric", default="cos", choices=["cos", "l1"])
    p.add_argument("--vmin", type=float, default=None, help="Por defecto 0.40 en cos, percentil 2 en l1")
    p.add_argument("--vmax", type=float, default=None, help="Por defecto 0.90 en cos, percentil 98 en l1")
    p.add_argument("--baseline", type=float, default=0.4291,
                   help="Referencia del predictor trivial, solo con --metric cos (0 = no dibujar)")
    p.add_argument("--frames-tira", type=int, default=8)
    p.add_argument("--gif-fps", type=float, default=8.0)
    p.add_argument("--gif-max-frames", type=int, default=0, help="0 = todos los frames")
    p.add_argument("--gif-colores", type=int, default=128)
    p.add_argument("--escala", type=int, default=6, help="Pixeles de salida por pixel de silueta")
    p.add_argument("--sin-tira", action="store_true")
    p.add_argument("--sin-gif", action="store_true")
    p.add_argument("--salida", default="visualizaciones")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    data_cfg, model_cfg = cfg['data_cfg'], cfg['model_cfg']
    og_root = Path(args.opengait_root).expanduser().resolve()

    net = construir_modelo(model_cfg, og_root, args.device)
    it = cargar_pesos(net, Path(args.checkpoint).expanduser(), args.device)

    from data.transform import get_transform
    trf = get_transform(cfg['evaluator_cfg']['transform'])[0]

    part_path = Path(args.partition).expanduser() if args.partition else \
        (og_root / data_cfg['dataset_partition']).resolve()
    pids = json.loads(part_path.read_text())[args.split]

    sil_root = Path(data_cfg['dataset_root'])
    teacher_root = Path(data_cfg['teacher_root'])
    stats = data_cfg.get('teacher_stats')
    if stats:
        st = np.load(stats)
        t_mean = torch.as_tensor(st['mean'], dtype=torch.float32, device=args.device)
        t_std = torch.as_tensor(np.maximum(st['std'], 1e-6), dtype=torch.float32, device=args.device)
    else:
        t_mean = t_std = None

    rels = elegir(sil_root, teacher_root, pids, args.seqs, args.conds, args.angulo)
    if not rels:
        sys.exit("Ninguna secuencia disponible: revisa --seqs / --conds / --angulo y el split")
    print(f"[split] {args.split}: {len(pids)} sujetos | secuencias: {', '.join(str(r) for r in rels)}")

    # ---- métricas de todas las secuencias antes de dibujar (la escala es común) ----
    datos = []
    for rel in rels:
        pkls = sorted((sil_root / rel).glob("*.pkl"))
        feat = teacher_root / rel / "features.npy"
        if not pkls or not feat.exists():
            print(f"  [SKIP] {rel}: falta silueta o features")
            continue
        sils = np.asarray(pickle.load(open(pkls[0], 'rb')))
        teacher = np.load(feat, mmap_mode='r')
        if teacher.ndim != 3 or teacher.shape[0] != len(sils):
            print(f"  [SKIP] {rel}: T maestro={teacher.shape[0]} vs siluetas={len(sils)}")
            continue
        if sils.shape[1] % PARTS:
            print(f"  [SKIP] {rel}: altura {sils.shape[1]} no es múltiplo de {PARTS}")
            continue
        cos, l1 = metricas(net, trf, sils, teacher, t_mean, t_std, args.device)
        datos.append((rel, sils, cos if args.metric == "cos" else l1))
        print(f"  {rel}: {len(sils)} frames | {args.metric} medio {datos[-1][2].mean():.4f}")

    if not datos:
        sys.exit("No se pudo procesar ninguna secuencia")

    todo = np.concatenate([d[2].ravel() for d in datos])
    if args.metric == "cos":
        vmin = args.vmin if args.vmin is not None else 0.40
        vmax = args.vmax if args.vmax is not None else 0.90
        ref = args.baseline if args.baseline > 0 else None
        nombre = "cos"
    else:
        vmin = args.vmin if args.vmin is not None else float(np.percentile(todo, 2))
        vmax = args.vmax if args.vmax is not None else float(np.percentile(todo, 98))
        ref, nombre = None, "L1"
    esc = Escala(vmin, vmax, invertir=(args.metric == "l1"))
    print(f"[escala] {nombre} en [{vmin:.3f}, {vmax:.3f}]"
          f"{' (invertida: claro = menor error)' if esc.invertir else ''}"
          f" | datos en [{todo.min():.3f}, {todo.max():.3f}]")

    salida = Path(args.salida).expanduser()
    salida.mkdir(parents=True, exist_ok=True)
    s = args.escala

    for rel, sils, vals in datos:
        T = len(sils)
        alto = sils.shape[1] * s
        serie = vals.mean(1)
        media = vals.mean(0)
        nom = "-".join(rel.parts)
        cab = f"{rel}  |  iteracion {it}  |  {nombre} por franja"

        if not args.sin_tira:
            idx = np.unique(np.linspace(0, T - 1, min(args.frames_tira, T)).astype(int))
            medio = sils[idx[len(idx) // 2]]
            paneles = [etiquetar(render_silueta(medio, media, esc, s, tenir=False), "silueta")]
            paneles += [etiquetar(render_silueta(sils[j], vals[j], esc, s), f"f{j}   {serie[j]:.3f}")
                        for j in idx]
            paneles += [etiquetar(render_silueta(medio, media, esc, s), "media secuencia"),
                        etiquetar(render_barras(media, esc, alto, 50 * s, ref), "media por franja"),
                        etiquetar(render_cbar(esc, alto), nombre)]
            sub = (f"{T} frames  |  franja = 4 filas de silueta (0 = cabeza)  |  "
                   f"media {vals.mean():.4f}" + (f"  |  trivial {ref:.4f}" if ref else ""))
            dst = salida / f"{nom}_tira.png"
            cv2.imwrite(str(dst), con_titulo(fila(paneles), cab, sub))
            print(f"  [tira] {dst}")

        if not args.sin_gif:
            jdx = range(T) if not args.gif_max_frames or T <= args.gif_max_frames else \
                np.unique(np.linspace(0, T - 1, args.gif_max_frames).astype(int))
            frames = []
            for j in jdx:
                paneles = [etiquetar(render_silueta(sils[j], media, esc, s, tenir=False), "silueta"),
                           etiquetar(render_silueta(sils[j], vals[j], esc, s), "destilacion"),
                           etiquetar(render_barras(vals[j], esc, alto, 50 * s, ref), f"{nombre} por franja"),
                           etiquetar(render_curva(serie, j, esc, alto, 62 * s, ref), "media por frame"),
                           etiquetar(render_cbar(esc, alto), nombre)]
                frames.append(con_titulo(fila(paneles), cab,
                                         f"frame {j + 1}/{T}  |  media del frame {serie[j]:.4f}"))
            guardar_gif(frames, salida / f"{nom}.gif", args.gif_fps, args.gif_colores)


if __name__ == "__main__":
    main()
