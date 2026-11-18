#!/usr/bin/env python3
"""
Evalúa reconocimiento de marcha en CASIA-B con el protocolo estándar de OpenGait y
reporta lo que el evaluador de OpenGait no da: Rank-5, mAP y el desglose por ángulo.

Protocolo (el de los papers de CASIA-B, idéntico a single_view_gallery_evaluation):
    galería  nm-01..04
    sondas   NM: nm-05/06    BG: bg-01/02    CL: cl-01/02
    Rank-k   para cada par (vista sonda, vista galería) se busca la sonda solo en la
             galería de esa vista; se excluye la diagonal (vista idéntica) y se
             promedia. Cada celda se redondea a 2 decimales como hace OpenGait, para
             que el Rank-1 coincida con el de los papers.
    mAP      cada sonda contra toda la galería de las demás vistas (también excluye
             la vista idéntica): pondera el ranking completo, no solo el primer acierto.

Embedding:
    fcs       salida de FCs (o de BNNecks si use_emb2), el que usa OpenGait al evaluar.
    backbone  HPP(TP(layer4)) sin cabeza: sirve para medir cuánta identidad trae un
              backbone que nunca se entrenó para reconocer (el de la fase 1, o uno
              aleatorio como piso), porque su cabeza FCs no está entrenada.

Uso:
    python data/evaluar_reconocimiento.py \
        --checkpoint ~/OpenGait/output/CASIA-B/DeepGaitV2/DeepGaitV2_distill/checkpoints/DeepGaitV2_distill-60000.pt \
        --cfg configs/fase2_distill.yaml --opengait-root ~/OpenGait --salida metricas.json

    # piso y punto de partida, sin entrenar reconocimiento
    python data/evaluar_reconocimiento.py --checkpoint aleatorio --embedding backbone ...
    python data/evaluar_reconocimiento.py --checkpoint backbone_fase1_20k_deepgaitv2.pt --embedding backbone ...
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

GALERIA = ["nm-01", "nm-02", "nm-03", "nm-04"]
SONDAS = {"NM": ["nm-05", "nm-06"], "BG": ["bg-01", "bg-02"], "CL": ["cl-01", "cl-02"]}
RANKS = (1, 5)
CAPAS = ("layer0", "layer1", "layer2", "layer3", "layer4")


# --------------------------------------------------------------------------- #
#  Modelo
# --------------------------------------------------------------------------- #
def construir_modelo(model_cfg, opengait_root, device, seed=0):
    """DeepGaitV2 sin BaseModel.__init__ (que exige DDP y data loaders)."""
    og = Path(opengait_root).expanduser().resolve() / "opengait"
    if not og.is_dir():
        sys.exit(f"No existe {og}: revisa --opengait-root")
    sys.path.insert(0, str(og))
    import torch.nn as nn
    from modeling.models.deepgaitv2 import DeepGaitV2

    if model_cfg["model"] != "DeepGaitV2":
        sys.exit(f"Se esperaba model: DeepGaitV2 en el yaml, no {model_cfg['model']}")
    torch.manual_seed(seed)
    net = DeepGaitV2.__new__(DeepGaitV2)
    nn.Module.__init__(net)
    net.build_network(model_cfg)
    net.init_parameters()                      # misma inicialización que un entrenamiento desde cero
    return net.to(device).eval()


def a_nombres_deepgaitv2(sd):
    """Acepta checkpoints de DeepGaitV2 o de StudentDistill (sin SetBlockWrapper)."""
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    if not any(k.startswith("proj_head.") or k.startswith("layer0.0.") for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith("proj_head."):
            continue
        head, _, rest = k.partition(".")
        out[f"{head}.forward_block.{rest}" if head in CAPAS else k] = v
    return out


def cargar(net, checkpoint, embedding, device):
    """Devuelve (iteración, descripción). 'aleatorio' deja la inicialización."""
    if checkpoint == "aleatorio":
        return 0, "aleatorio"
    ck = torch.load(Path(checkpoint).expanduser(), map_location=device, weights_only=False)
    sd = a_nombres_deepgaitv2(ck.get("model", ck))
    res = net.load_state_dict(sd, strict=False)
    faltan_bb = [k for k in res.missing_keys if k.split(".")[0] in CAPAS]
    if faltan_bb:
        sys.exit(f"Faltan {len(faltan_bb)} pesos del backbone, p.ej. {faltan_bb[:4]}: "
                 "el checkpoint no corresponde a esta arquitectura")
    if res.missing_keys and embedding == "fcs":
        sys.exit(f"Faltan pesos de la cabeza ({res.missing_keys[:4]}); usa --embedding backbone")
    if res.unexpected_keys:
        print(f"[warn] {len(res.unexpected_keys)} claves sobrantes ignoradas: {res.unexpected_keys[:4]}")
    it = int(ck.get("iteration", 0) or 0)
    if embedding == "fcs" and "optimizer" not in ck:
        print("[warn] el checkpoint no trae optimizador: parece un backbone exportado de la fase 1, "
              "cuya cabeza FCs no se entrenó para reconocer. Para él usa --embedding backbone.")
    return it, str(checkpoint)


# --------------------------------------------------------------------------- #
#  Extracción de embeddings
# --------------------------------------------------------------------------- #
def secuencias(sil_root, pids):
    for pid in pids:
        for d in sorted(p for p in (sil_root / pid).glob("*/*") if p.is_dir()):
            pkls = sorted(d.glob("*.pkl"))
            if pkls:
                yield d.relative_to(sil_root), pkls[0]


@torch.no_grad()
def embedding_secuencia(net, x, embedding, chunk):
    """x: [s, h, w]. El TP es un máximo sobre frames y BN está en eval, así que
    procesar los frames por trozos y quedarse con el máximo es exacto."""
    tp = None
    for i in range(0, len(x), chunk):
        z = x[i:i + chunk][None, None]                                  # [1, 1, s, h, w]
        for capa in CAPAS:
            z = getattr(net, capa)(z)
        z = z.max(2)[0]                                                 # [1, c, h, w]
        tp = z if tp is None else torch.maximum(tp, z)
    f = net.HPP(tp)                                                     # [1, c, p]
    if embedding == "fcs":
        f = net.FCs(f)
        if getattr(net, "inference_use_emb2", False):
            f = net.BNNecks(f)[0]
    return f[0].float().cpu().numpy()


def extraer(net, trf, seqs, embedding, device, chunk, fp16):
    feats, pid, cond, vista = [], [], [], []
    for n, (rel, pkl) in enumerate(seqs, 1):
        with open(pkl, "rb") as fh:
            sils = np.asarray(pickle.load(fh))
        if sils.ndim != 3 or len(sils) == 0:
            print(f"  [SKIP] {rel}: forma {sils.shape}")
            continue
        x = torch.from_numpy(np.stack([trf(s) for s in sils]).astype(np.float32)).to(device)
        with torch.autocast(device_type=x.device.type, enabled=fp16):
            feats.append(embedding_secuencia(net, x, embedding, chunk))
        pid.append(rel.parts[0]); cond.append(rel.parts[1]); vista.append(rel.parts[2])
        if n % 500 == 0:
            print(f"  {n} secuencias", flush=True)
    return np.stack(feats), np.array(pid), np.array(cond), np.array(vista)


# --------------------------------------------------------------------------- #
#  Métricas
# --------------------------------------------------------------------------- #
def distancias(P, G, metric, device):
    """Igual que cuda_dist de OpenGait: distancia por franja, promediada. P [n,c,p], G [m,c,p]."""
    x = torch.from_numpy(P).to(device, torch.float64)
    y = torch.from_numpy(G).to(device, torch.float64)
    if metric == "cos":
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        return (1 - torch.einsum("ncp,mcp->nm", x, y) / x.shape[2]).cpu().numpy()
    d = torch.zeros(len(x), len(y), dtype=torch.float64, device=device)
    for i in range(x.shape[2]):
        a, b = x[:, :, i], y[:, :, i]
        d += torch.sqrt(torch.relu((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T))
    return (d / x.shape[2]).cpu().numpy()


def average_precision(aciertos):
    """aciertos: vector 0/1 ordenado por distancia creciente."""
    pos = np.flatnonzero(aciertos)
    if len(pos) == 0:
        return np.nan
    return float(np.mean(np.arange(1, len(pos) + 1) / (pos + 1)))


def evaluar(feat, pid, cond, vista, metric, device):
    vistas = sorted(np.unique(vista))
    V = len(vistas)
    gal = np.isin(cond, GALERIA)
    res = {"vistas": vistas}
    for tipo, sondas in SONDAS.items():
        son = np.isin(cond, sondas)
        acc = np.full((len(RANKS), V, V), np.nan)
        for i, vp in enumerate(vistas):
            pm = son & (vista == vp)
            if not pm.any():
                continue
            for j, vg in enumerate(vistas):
                gm = gal & (vista == vg)
                if not gm.any():
                    continue
                d = distancias(feat[pm], feat[gm], metric, device)
                orden = np.argsort(d, 1, kind="stable")[:, :max(RANKS)]
                cum = np.cumsum(pid[gm][orden] == pid[pm][:, None], 1) > 0
                for k, r in enumerate(RANKS):
                    acc[k, i, j] = np.round(100 * cum[:, min(r, cum.shape[1]) - 1].mean(), 2)
        fuera = ~np.eye(V, dtype=bool)
        por_vista = {f"R{r}_por_vista": [float(np.nanmean(acc[k, i][fuera[i]])) for i in range(V)]
                     for k, r in enumerate(RANKS)}

        # mAP: toda la galería de las demás vistas
        idx_p = np.flatnonzero(son)
        d = distancias(feat[idx_p], feat[gal], metric, device)
        aps = []
        for q, ip in enumerate(idx_p):
            m = vista[gal] != vista[ip]
            orden = np.argsort(d[q, m], kind="stable")
            aps.append(average_precision(pid[gal][m][orden] == pid[ip]))

        res[tipo] = {**{f"R{r}": float(np.nanmean(por_vista[f"R{r}_por_vista"])) for r in RANKS},
                     "mAP": 100 * float(np.nanmean(aps)),
                     **por_vista,
                     "R1_matriz": acc[0].tolist(),
                     "n_sondas": int(son.sum())}
    res["media"] = {m: float(np.mean([res[t][m] for t in SONDAS]))
                    for m in [f"R{r}" for r in RANKS] + ["mAP"]}
    return res


def imprimir(r, cab):
    print(f"\n=== {cab} ===")
    cols = [f"R{x}" for x in RANKS] + ["mAP"]
    print(f"{'':<7}" + "".join(f"{c:>9}" for c in cols))
    for t in list(SONDAS) + ["media"]:
        print(f"{t:<7}" + "".join(f"{r[t][c]:>9.2f}" for c in cols))
    print("Rank-1 por ángulo de la sonda (excluye vista idéntica):")
    print(f"{'':<7}" + "".join(f"{v:>7}" for v in r["vistas"]))
    for t in SONDAS:
        print(f"{t:<7}" + "".join(f"{x:>7.1f}" for x in r[t]["R1_por_vista"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Ruta .pt o 'aleatorio'")
    p.add_argument("--cfg", default="configs/fase2_distill.yaml")
    p.add_argument("--opengait-root", default="~/OpenGait")
    p.add_argument("--partition", default="", help="Por defecto, el del yaml resuelto contra --opengait-root")
    p.add_argument("--split", default="TEST_SET", choices=["TEST_SET", "TRAIN_SET"])
    p.add_argument("--embedding", default="fcs", choices=["fcs", "backbone"])
    p.add_argument("--metric", default="", choices=["", "euc", "cos"], help="Por defecto, la del yaml")
    p.add_argument("--max-seqs", type=int, default=0, help="Solo para pruebas rápidas")
    p.add_argument("--chunk-frames", type=int, default=256)
    p.add_argument("--fp16", action="store_true", help="autocast como OpenGait (enable_float16)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--salida", default="", help="JSON de métricas")
    p.add_argument("--guardar-embeddings", default="", help="npz con embeddings y etiquetas")
    p.add_argument("--etiqueta", default="", help="Nombre del brazo, se guarda en el JSON")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    data_cfg, model_cfg = cfg["data_cfg"], cfg["model_cfg"]
    metric = args.metric or cfg.get("evaluator_cfg", {}).get("metric", "euc")
    og_root = Path(args.opengait_root).expanduser().resolve()

    net = construir_modelo(model_cfg, og_root, args.device)
    it, origen = cargar(net, args.checkpoint, args.embedding, args.device)
    print(f"[ckpt] {origen} | iteración {it} | embedding {args.embedding} | métrica {metric}")

    from data.transform import get_transform
    trf = get_transform(cfg["evaluator_cfg"]["transform"])[0]

    part = Path(args.partition).expanduser() if args.partition else \
        (og_root / data_cfg["dataset_partition"]).resolve()
    pids = json.loads(part.read_text())[args.split]
    seqs = list(secuencias(Path(data_cfg["dataset_root"]), pids))
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    if not seqs:
        sys.exit("No hay secuencias: revisa dataset_root y la partición")
    print(f"[split] {args.split}: {len(pids)} sujetos, {len(seqs)} secuencias")

    feat, pid, cond, vista = extraer(net, trf, seqs, args.embedding, args.device,
                                     args.chunk_frames, args.fp16)
    if not np.isin(cond, GALERIA).any():
        sys.exit("No hay secuencias de galería (nm-01..04) en el split")
    r = evaluar(feat, pid, cond, vista, metric, args.device)
    r.update(checkpoint=origen, iteracion=it, embedding=args.embedding, metrica=metric,
             split=args.split, secuencias=int(len(feat)), etiqueta=args.etiqueta)
    imprimir(r, f"Reconocimiento en {args.split} | iteración {it} | embedding {args.embedding}")

    if args.salida:
        Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
        Path(args.salida).write_text(json.dumps(r, indent=1))
        print(f"\n[json] {args.salida}")
    if args.guardar_embeddings:
        np.savez_compressed(args.guardar_embeddings, feat=feat.astype(np.float16),
                            pid=pid, cond=cond, vista=vista)
        print(f"[npz] {args.guardar_embeddings}")


if __name__ == "__main__":
    main()
