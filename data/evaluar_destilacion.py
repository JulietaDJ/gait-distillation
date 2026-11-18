#!/usr/bin/env python3
"""
Evalúa la destilación sobre identidades no vistas y exporta el backbone sin proj_head.

El entrenamiento solo informa cos_sim sobre TRAIN_SET, lo que no dice si el mapeo
silueta -> maestro generaliza a sujetos nuevos. Este script recorre el split indicado
(TEST_SET por defecto) y reporta:

    cos_sim        similitud coseno media estudiante/maestro, por token [frame, franja]
    loss_l1        Smooth L1 medio (misma beta que la pérdida de entrenamiento)
    baseline       cos_sim de un predictor trivial = media por franja del propio split
    gap cerrado    (cos_sim - baseline) / (1 - baseline): cuánto aporta el estudiante
    CKA lineal     similitud de estructura entre ambos espacios de features
    cos por franja las 16 franjas por separado (detecta franjas que no se aprenden)

Todas las cifras se calculan en una sola pasada y en float64.

Uso:
    python data/evaluar_destilacion.py \
        --checkpoint ~/OpenGait/output/CASIA-B/StudentDistill/StudentDistill_Phase1/checkpoints/StudentDistill_Phase1-10000.pt \
        --cfg configs/student_distill_casiab.yaml \
        --opengait-root ~/OpenGait \
        --export backbone_fase1
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# --------------------------------------------------------------------------- #
#  Modelo
# --------------------------------------------------------------------------- #
def construir_modelo(model_cfg, opengait_root, device):
    """Instancia StudentDistill sin pasar por BaseModel.__init__ (que exige DDP/CUDA)."""
    og = Path(opengait_root).expanduser().resolve() / "opengait"
    if not og.is_dir():
        sys.exit(f"No existe {og}: revisa --opengait-root")
    sys.path.insert(0, str(og))
    import torch.nn as nn
    from modeling.models.student_distill import StudentDistill

    net = StudentDistill.__new__(StudentDistill)
    nn.Module.__init__(net)
    net.build_network(model_cfg)
    return net.to(device).eval()


def cargar_pesos(net, checkpoint, device):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ck.get('model', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    res = net.load_state_dict(sd, strict=False)
    if res.missing_keys:
        sys.exit(f"Faltan {len(res.missing_keys)} pesos en el checkpoint, p.ej. {res.missing_keys[:5]}")
    if res.unexpected_keys:
        print(f"[warn] {len(res.unexpected_keys)} claves sobrantes ignoradas: {res.unexpected_keys[:5]}")
    print(f"[ckpt] iteración {ck.get('iteration', '?')} | {checkpoint}")
    return ck.get('iteration', 0)


def exportar(net, destino, class_num):
    """Guarda el backbone sin proj_head, en las dos convenciones de nombres útiles."""
    sd = {k: v for k, v in net.state_dict().items() if not k.startswith('proj_head.')}
    destino = Path(destino)

    p1 = destino.with_name(destino.name + "_student.pt")
    torch.save({'model': sd, 'iteration': 0}, p1)

    # DeepGaitV2 envuelve layer0..layer4 en SetBlockWrapper -> prefijo 'forward_block.'
    remap = {}
    for k, v in sd.items():
        if k.split('.')[0] in ('layer0', 'layer1', 'layer2', 'layer3', 'layer4'):
            head, rest = k.split('.', 1)
            remap[f"{head}.forward_block.{rest}"] = v
        else:
            remap[k] = v
    p2 = destino.with_name(destino.name + "_deepgaitv2.pt")
    torch.save({'model': remap, 'iteration': 0}, p2)

    print(f"[export] {p1}  (StudentDistill / fase 2 con recognition_losses)")
    print(f"[export] {p2}  (DeepGaitV2 mode 2d, class_num={class_num})")


# --------------------------------------------------------------------------- #
#  Acumuladores
# --------------------------------------------------------------------------- #
class Acumulador:
    """Métricas por token en una pasada. Todo en float64.

    El baseline por franja se obtiene sin segunda pasada: si m_p es la media de los
    tokens del maestro en la franja p y A_p la media de esos tokens normalizados,
        mean_t cos(t, m_p) = (A_p . m_p) / |m_p|
    """

    def __init__(self, dim, parts, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.dim, self.parts = dim, parts
        self.cos_sum, self.l1_sum, self.n = 0.0, 0.0, 0
        self.cos_part, self.n_part = z(parts), z(parts)
        self.t_sum, self.t_unit = z(parts, dim), z(parts, dim)
        self.XtX, self.YtY, self.YtX = z(dim, dim), z(dim, dim), z(dim, dim)
        self.sx, self.sy = z(dim), z(dim)

    def add(self, s, t):
        """s, t: [n, parts, dim] (estudiante, maestro ya estandarizado)."""
        s, t = s.double(), t.double()
        cos = F.cosine_similarity(s, t, dim=-1, eps=1e-8)          # [n, parts]
        self.cos_sum += float(cos.sum())
        self.l1_sum += float(F.smooth_l1_loss(s, t, beta=1.0, reduction='sum')) / self.dim
        self.n += cos.numel()
        self.cos_part += cos.sum(0)
        self.n_part += cos.shape[0]

        self.t_sum += t.sum(0)
        self.t_unit += (t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)).sum(0)

        X = s.reshape(-1, self.dim)
        Y = t.reshape(-1, self.dim)
        self.XtX += X.T @ X
        self.YtY += Y.T @ Y
        self.YtX += Y.T @ X
        self.sx += X.sum(0)
        self.sy += Y.sum(0)

    def cka(self):
        n = float(self.n)
        mx, my = self.sx / n, self.sy / n
        Xc = self.XtX - n * torch.outer(mx, mx)
        Yc = self.YtY - n * torch.outer(my, my)
        C = self.YtX - n * torch.outer(my, mx)
        den = Xc.norm() * Yc.norm()
        return float(C.norm() ** 2 / den) if den > 0 else float('nan')

    def baseline(self):
        m = self.t_sum / self.n_part[:, None]                       # [parts, dim]
        a = self.t_unit / self.n_part[:, None]
        per_part = (a * m).sum(-1) / m.norm(dim=-1).clamp_min(1e-12)
        return float(per_part.mean())

    def resumen(self):
        cos = self.cos_sum / self.n
        base = self.baseline()
        return {
            'cos_sim': cos,
            'loss_l1': self.l1_sum / self.n,
            'baseline': base,
            'gap': (cos - base) / (1.0 - base) if base < 1.0 else float('nan'),
            'cka': self.cka(),
            'cos_part': (self.cos_part / self.n_part).cpu().numpy(),
            'tokens': self.n,
        }


# --------------------------------------------------------------------------- #
#  Recorrido del split
# --------------------------------------------------------------------------- #
def secuencias(sil_root, teacher_root, pids):
    for pid in pids:
        for d in sorted(p for p in (teacher_root / pid).glob("*/*") if (p / "features.npy").exists()):
            rel = d.relative_to(teacher_root)
            sil_dir = sil_root / rel
            pkls = sorted(sil_dir.glob("*.pkl"))
            if pkls:
                yield rel, pkls[0], d / "features.npy"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cfg", default="configs/student_distill_casiab.yaml")
    p.add_argument("--opengait-root", default="~/OpenGait")
    p.add_argument("--partition", default="", help="Por defecto, el del yaml resuelto contra --opengait-root")
    p.add_argument("--split", default="TEST_SET", choices=["TEST_SET", "TRAIN_SET"])
    p.add_argument("--max-frames-seq", type=int, default=20,
                   help="Frames equiespaciados por secuencia (0 = todos)")
    p.add_argument("--max-seqs", type=int, default=0, help="Limitar nº de secuencias (0 = todas)")
    p.add_argument("--batch-frames", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--export", default="", help="Prefijo de salida del backbone sin proj_head")
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
    print(f"[split] {args.split}: {len(pids)} sujetos | partición {part_path}")

    sil_root = Path(data_cfg['dataset_root'])
    teacher_root = Path(data_cfg['teacher_root'])
    stats = data_cfg.get('teacher_stats')
    if stats:
        st = np.load(stats)
        t_mean = torch.as_tensor(st['mean'], dtype=torch.float32, device=args.device)
        t_std = torch.as_tensor(np.maximum(st['std'], 1e-6), dtype=torch.float32, device=args.device)
    else:
        t_mean = t_std = None

    acc = Acumulador(model_cfg.get('teacher_dim', 768), 16, args.device)
    n_ok = n_skip = 0

    with torch.no_grad():
        for rel, pkl, feat_path in secuencias(sil_root, teacher_root, pids):
            if args.max_seqs and n_ok >= args.max_seqs:
                break
            sils = np.asarray(pickle.load(open(pkl, 'rb')))
            teacher = np.load(feat_path, mmap_mode='r')
            if teacher.ndim != 3 or teacher.shape[0] != len(sils):
                print(f"  [SKIP] {rel}: T maestro={teacher.shape[0]} vs siluetas={len(sils)}")
                n_skip += 1
                continue

            T = len(sils)
            idx = np.arange(T) if not args.max_frames_seq or T <= args.max_frames_seq else \
                np.unique(np.linspace(0, T - 1, args.max_frames_seq).astype(int))

            for i in range(0, len(idx), args.batch_frames):
                sub = idx[i:i + args.batch_frames]
                x = np.stack([trf(sils[j]) for j in sub]).astype(np.float32)
                x = torch.from_numpy(x).to(args.device)[:, None]              # [n, 1, H, W]
                out = net.layer4(net.layer3(net.layer2(net.layer1(net.layer0(x)))))
                s = net.proj_head(net.HPP(out).permute(0, 2, 1))              # [n, 16, 768]

                t = torch.from_numpy(np.asarray(teacher[sub], dtype=np.float32)).to(args.device)
                if t_mean is not None:
                    t = (t - t_mean) / t_std
                acc.add(s, t)

            n_ok += 1
            if n_ok % 200 == 0:
                print(f"  {n_ok} secuencias | cos_sim parcial = {acc.cos_sum / acc.n:.4f}")

    if acc.n == 0:
        sys.exit("No se evaluó ningún token: revisa dataset_root/teacher_root y el split")

    r = acc.resumen()
    print(f"\n=== Destilación en {args.split} (iteración {it}) ===")
    print(f"secuencias     : {n_ok} evaluadas, {n_skip} descartadas")
    print(f"tokens         : {r['tokens']}")
    print(f"cos_sim        : {r['cos_sim']:.4f}")
    print(f"baseline franja: {r['baseline']:.4f}")
    print(f"gap cerrado    : {100 * r['gap']:.1f} %")
    print(f"loss_l1        : {r['loss_l1']:.4f}")
    print(f"CKA lineal     : {r['cka']:.4f}")
    cp = r['cos_part']
    print("cos por franja (0 = arriba):")
    print("  " + "  ".join(f"{v:.3f}" for v in cp))
    print(f"  min {cp.min():.3f} (franja {int(cp.argmin())}) | max {cp.max():.3f} (franja {int(cp.argmax())})")

    if args.export:
        exportar(net, args.export, model_cfg['SeparateBNNecks']['class_num'])


if __name__ == "__main__":
    main()
