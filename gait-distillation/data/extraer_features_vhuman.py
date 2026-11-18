#!/usr/bin/env python3
"""
Extrae las features del maestro VHuman3D en formato compacto.

Salida por secuencia (<out>/<pid>/<cond>/<view>/):
    features.npy : float16 [T, 16, 768]   (antes float32 [T, 16, 16, 768], ~32x más grande)
    frames.txt   : nombres de los T frames, en el mismo orden que el .pkl de siluetas

Entrada al modelo (según model_builder.py y demo.py de VHuman3D):
    - NHWC, RGB, float en [0, 1]; forward_backbone hace movedim y preprocessing
      redimensiona a 518x518 y normaliza internamente.
    - preprocessing NO conserva la proporción: la imagen se rellena antes a un
      cuadrado de pad_factor x lado mayor, igual que pad_cropped_image() del demo.

Rejilla 3D [G0, G1, G2, C] (convención de plot_3d del demo):
    G0 = horizontal, G1 = vertical (índice 0 arriba), G2 = profundidad.
    Por defecto: max sobre G2 (--depth-axis 2) y mean sobre G0 (--reduce-axis 0),
    quedan 16 franjas verticales como en la HPP del estudiante.

Robustez:
    - Si falla un solo frame, la secuencia completa se descarta (evita desalinear índices).
    - Si se da --sil-root, solo se procesan secuencias cuyos frames coinciden con las siluetas.
    - Reanudable: salta secuencias ya extraídas salvo --overwrite.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def cargar_modelo(config_path, weights_path, device, allow_missing=False):
    from vhuman3d.architectures import model_builder
    model = model_builder(config_path).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    res = model.load_state_dict(state, strict=False)

    # La DINOv2 congelada no viaja en model_clean.pt: la aporta dinov2_vitb14_reg() desde torch hub.
    miss_bb = [k for k in res.missing_keys if k.startswith("backbone.")]
    miss_other = [k for k in res.missing_keys if not k.startswith("backbone.")]
    print(f"[load] missing backbone.*: {len(miss_bb)} | missing otras: {len(miss_other)} "
          f"| unexpected: {len(res.unexpected_keys)}")
    for k in miss_other[:20]:
        print(f"   missing   : {k}")
    for k in res.unexpected_keys[:20]:
        print(f"   unexpected: {k}")
    if miss_bb:
        ckpts = sorted((Path(torch.hub.get_dir()) / "checkpoints").glob("dinov2_vitb14_reg*"))
        print(f"[load] DINOv2 desde torch hub: {[c.name for c in ckpts] or 'NO ENCONTRADO en caché'}")
    if miss_other and not allow_missing:
        sys.exit("[load] Hay pesos del decoder 3D sin cargar: el maestro estaría parcialmente "
                 "aleatorio. Revisa el checkpoint o usa --allow-missing si es intencionado.")
    model.eval()
    return model


def pad_cuadrado(img, pad_factor):
    """Réplica de pad_cropped_image() del demo: centra en un cuadrado de pad_factor x lado mayor."""
    h, w = img.shape[:2]
    side = int(max(h, w) * pad_factor)
    canvas = np.zeros((side, side, img.shape[2]), dtype=img.dtype)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img
    return canvas


def preprocesar(rgb_list, pad_factor, device):
    """Lista de RGB uint8 [H, W, 3] -> tensor NHWC float [B, S, S, 3] en [0, 1]."""
    imgs = [img if img.shape[0] == img.shape[1] else pad_cuadrado(img, pad_factor) for img in rgb_list]
    side = max(i.shape[0] for i in imgs)
    imgs = [i if i.shape[0] == side else cv2.resize(i, (side, side), interpolation=cv2.INTER_LINEAR)
            for i in imgs]
    return torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0).to(device)


@torch.no_grad()
def volumen_3d(model, x):
    """Tokens del decoder 3D (API oficial get_features) -> [B, G0, G1, G2, C]."""
    tokens = model.get_features(x)                                  # [B, G0*G1*G2, C]
    return tokens.reshape(tokens.shape[0], *model.init_3d_grid_shape, model.embedding_dim_3d)


def reducir(vol, depth_axis, reduce_axis):
    """[B, G0, G1, G2, C] -> [B, G_restante, C]."""
    d, r = depth_axis + 1, reduce_axis + 1
    vol = vol.amax(dim=d)
    r = r - 1 if r > d else r
    return vol.mean(dim=r)


@torch.no_grad()
def check(model, device, rgb_root, pads):
    """Diagnóstico: extensión del cuerpo predicho por eje y su posición en las 16 franjas verticales."""
    print(f"[check] init_3d_grid_shape = {tuple(model.init_3d_grid_shape)}, "
          f"embedding_dim_3d = {model.embedding_dim_3d}")
    seqs = sorted(p for p in rgb_root.glob("*/nm-01/090") if any(p.glob("*.png"))) or \
        sorted(p for p in rgb_root.glob("*/*/*") if p.is_dir() and any(p.glob("*.png")))
    frames = sorted(seqs[0].glob("*.png"))
    frames = [frames[i] for i in np.linspace(0, len(frames) - 1, 8).astype(int)]
    rgb = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in frames]
    print(f"[check] secuencia: {seqs[0]} | frame RGB guardado: {rgb[0].shape}")

    for pad in pads:
        x = preprocesar(rgb, pad, device)
        segm, _ = model(x)                                          # [B, 11, 64, 64, 64]
        occ = segm.argmax(dim=1) != 0                               # [B, 64, 64, 64]
        print(f"[check] pad_factor={pad}: entrada {tuple(x.shape)} | vóxeles ocupados/frame = "
              f"{occ.flatten(1).sum(1).float().mean():.0f}")
        for ax in range(3):
            dims = tuple(d for d in (1, 2, 3) if d != ax + 1)
            prof = occ.any(dim=dims[1]).any(dim=dims[0])             # [B, 64]
            lo = [int(p.nonzero()[0]) if p.any() else -1 for p in prof]
            hi = [int(p.nonzero()[-1]) if p.any() else -1 for p in prof]
            print(f"         eje {ax}: extensión media {np.mean(np.subtract(hi, lo)) + 1:.1f}/64 vóxeles"
                  f" | rango medio [{np.mean(lo):.1f}, {np.mean(hi):.1f}]"
                  f" -> franjas 16 [{np.mean(lo) / 4:.1f}, {np.mean(hi) / 4:.1f}]")
    print("[check] Esperado: eje 1 = vertical (mayor extensión). Si las franjas del eje 1 no cubren "
          "~[0, 15], el cuerpo no ocupa toda la altura de la rejilla y conviene ajustar --pad-factor.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-root", default="/opt3/data/julieta/CASIA-B-yolo/rgb/")
    p.add_argument("--sil-root", default="/opt3/data/julieta/CASIA-B-yolo/silhouettes/",
                   help="Para verificar que los frames coinciden con las siluetas ('' para omitir)")
    p.add_argument("--output", default="/opt3/data/julieta/vhuman_features_v2")
    p.add_argument("--vhuman-root", default="/opt3/data/julieta/vhuman3d/vhuman3d_deploy")
    p.add_argument("--config", default="vhuman3d/architectures/configs/vhuman3d_vitb.json")
    p.add_argument("--weights", default="runs/run_1778669751/model_clean.pt")
    p.add_argument("--max-sujetos", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Cross-attention 4096x1369 por frame: bajar si la GPU compartida va justa")
    p.add_argument("--pad-factor", type=float, default=1.2,
                   help="Relleno a cuadrado como el demo (1.0-1.5). Se omite si el RGB ya es cuadrado")
    p.add_argument("--depth-axis", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--reduce-axis", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check", action="store_true", help="Diagnóstico sobre 8 frames y salir")
    args = p.parse_args()
    if args.depth_axis == args.reduce_axis:
        sys.exit("--depth-axis y --reduce-axis deben ser distintos")

    rgb_root = Path(args.rgb_root).resolve()
    sil_root = Path(args.sil_root).resolve() if args.sil_root else None
    out_root = Path(args.output).resolve()

    sys.path.insert(0, args.vhuman_root)
    os.chdir(args.vhuman_root)  # config y weights son relativos a vhuman3d_deploy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")  # igual que demo.py
    print(f"Cargando modelo en {device}...")
    model = cargar_modelo(args.config, args.weights, device, args.allow_missing)

    if args.check:
        check(model, device, rgb_root, sorted({1.0, args.pad_factor}))
        return

    out_root.mkdir(parents=True, exist_ok=True)
    sujetos = sorted(d for d in rgb_root.iterdir() if d.is_dir())
    if args.max_sujetos > 0:
        sujetos = sujetos[:args.max_sujetos]

    n_ok = n_skip = n_fail = 0
    for sujeto in sujetos:
        for seq in sorted(p for p in sujeto.glob("*/*") if p.is_dir()):
            rel = seq.relative_to(rgb_root)
            out_dir = out_root / rel
            if (out_dir / "features.npy").exists() and not args.overwrite:
                n_skip += 1
                continue

            frames = sorted(seq.glob("*.png"))
            if not frames:
                continue
            if sil_root is not None:
                sil_names = sorted(f.name for f in (sil_root / rel).glob("*.png"))
                if sil_names != [f.name for f in frames]:
                    print(f"  [SKIP] {rel}: frames RGB y siluetas no coinciden")
                    n_fail += 1
                    continue

            try:
                feats = []
                for i in range(0, len(frames), args.batch_size):
                    rgb = []
                    for f in frames[i:i + args.batch_size]:
                        img = cv2.imread(str(f))
                        if img is None:
                            raise IOError(f"no se pudo leer {f.name}")
                        rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    vol = volumen_3d(model, preprocesar(rgb, args.pad_factor, device))
                    feats.append(reducir(vol, args.depth_axis, args.reduce_axis).half().cpu())
                feats = torch.cat(feats).numpy()                       # [T, 16, 768] float16
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {rel}: {e} -> secuencia descartada")
                n_fail += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / "features.tmp.npy"
            np.save(tmp, feats)
            tmp.replace(out_dir / "features.npy")                     # escritura atómica
            (out_dir / "frames.txt").write_text("\n".join(f.name for f in frames) + "\n")
            n_ok += 1
            print(f"  {rel}: {feats.shape}")

    print(f"\n=== Resumen === OK: {n_ok} | ya existentes: {n_skip} | descartadas: {n_fail}")


if __name__ == "__main__":
    main()
