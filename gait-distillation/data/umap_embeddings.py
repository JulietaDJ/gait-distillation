#!/usr/bin/env python3
"""
Proyecta con UMAP los embeddings del estudiante y del maestro en el mismo plano.

El coseno y el CKA resumen en un número si los dos espacios se parecen. Esta
proyección enseña *cómo* se parecen: se ajusta un único UMAP sobre la unión de
los dos conjuntos de embeddings, de modo que ambos caen en el mismo sistema de
coordenadas y la comparación es directa. Si el estudiante hubiera aprendido una
representación desplazada respecto al maestro -- cosa que un coseno alto no
descarta, porque es invariante a la norma -- las dos nubes saldrían separadas.

Genera tres vistas:

    umap_global.png      las 16 franjas juntas, coloreadas por franja y por fuente.
                         Que las franjas se separen no demostraría nada: la HPP le da
                         al estudiante la identidad de la franja gratis. Y puede que
                         ni siquiera se separen, porque el baseline de 0.43 implica que
                         la variación *dentro* de una franja supera a la que hay *entre*
                         franjas. Lo que sí se lee aquí es si las dos fuentes ocupan
                         las mismas regiones del plano.
    umap_franja_NN.png   una sola franja, que elimina esa estructura trivial y deja
                         ver lo que varía *dentro* de la banda: fuente, ángulo y
                         condición de marcha.
    umap_pares.png       segmentos maestro -> estudiante del mismo token. Segmento
                         corto = predicción cercana.

Junto a las figuras se informan dos cifras que hacen comprobable lo que la
proyección sugiere, porque UMAP puede inventar vecindarios que no existen:

    solapamiento k-NN    de los k vecinos de un token en el espacio del maestro,
                         cuántos siguen siéndolo en el del estudiante.
    rango de recuperación  buscando en el maestro con el embedding del estudiante,
                         en qué posición queda el token correcto.

Uso:
    python data/umap_embeddings.py \
        --checkpoint ~/OpenGait/output/CASIA-B/StudentDistill/StudentDistill_Phase1/checkpoints/StudentDistill_Phase1-10000.pt \
        --cfg configs/student_distill_casiab.yaml \
        --opengait-root ~/OpenGait \
        --salida umap
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from evaluar_destilacion import construir_modelo, cargar_pesos, secuencias

PARTS = 16
C_MAESTRO = "#3B6EA5"
C_ESTUDIANTE = "#D1702F"
C_COND = {"nm": "#4C9F70", "bg": "#B5523C", "cl": "#6B5B95"}
NOMBRE_COND = {"nm": "nm (normal)", "bg": "bg (mochila)", "cl": "cl (abrigo)"}


# --------------------------------------------------------------------------- #
#  Recolección de embeddings
# --------------------------------------------------------------------------- #
def muestrear(sil_root, teacher_root, pids, n_seqs, conds, angulos, seed):
    """Submuestra secuencias al azar; con 5499 disponibles sale repartido solo."""
    todas = []
    for rel, pkl, feat in secuencias(sil_root, teacher_root, pids):
        cond, vista = rel.parts[1], rel.parts[2]
        if conds and cond[:2] not in conds:
            continue
        if angulos and vista not in angulos:
            continue
        todas.append((rel, pkl, feat))
    if not todas:
        sys.exit("Ninguna secuencia pasa los filtros --conds/--angulos")
    rng = np.random.default_rng(seed)
    sel = sorted(rng.permutation(len(todas))[:n_seqs]) if n_seqs and n_seqs < len(todas) \
        else range(len(todas))
    return [todas[i] for i in sel]


@torch.no_grad()
def recolectar(net, trf, seqs, t_mean, t_std, frames_seq, device, batch):
    """Devuelve S, T [N, 768] y los metadatos alineados token a token."""
    S, T = [], []
    m_pid, m_cond, m_vista, m_frame, m_franja, m_seq = [], [], [], [], [], []
    n_skip = 0

    for i_seq, (rel, pkl, feat_path) in enumerate(seqs):
        with open(pkl, "rb") as fh:
            sils = np.asarray(pickle.load(fh))
        teacher = np.load(feat_path, mmap_mode="r")
        if teacher.ndim != 3 or teacher.shape[0] != len(sils):
            n_skip += 1
            continue

        n = len(sils)
        idx = np.unique(np.linspace(0, n - 1, min(frames_seq, n)).astype(int))
        pid, cond, vista = rel.parts[0], rel.parts[1], rel.parts[2]

        for i in range(0, len(idx), batch):
            sub = idx[i:i + batch]
            x = np.stack([trf(sils[j]) for j in sub]).astype(np.float32)
            x = torch.from_numpy(x).to(device)[:, None]
            out = net.layer4(net.layer3(net.layer2(net.layer1(net.layer0(x)))))
            s = net.proj_head(net.HPP(out).permute(0, 2, 1))              # [n, 16, 768]

            t = torch.from_numpy(np.asarray(teacher[sub], dtype=np.float32)).to(device)
            if t_mean is not None:
                t = (t - t_mean) / t_std

            S.append(s.reshape(-1, s.shape[-1]).float().cpu().numpy())
            T.append(t.reshape(-1, t.shape[-1]).float().cpu().numpy())
            for j in sub:
                m_pid += [pid] * PARTS
                m_cond += [cond] * PARTS
                m_vista += [vista] * PARTS
                m_frame += [int(j)] * PARTS
                m_seq += [i_seq] * PARTS
                m_franja += list(range(PARTS))

        if (i_seq + 1) % 50 == 0:
            print(f"  {i_seq + 1}/{len(seqs)} secuencias")

    if not S:
        sys.exit("No se recolectó ningún embedding: revisa dataset_root/teacher_root")

    meta = {
        "pid": np.array(m_pid), "cond": np.array(m_cond), "vista": np.array(m_vista),
        "frame": np.array(m_frame, np.int32), "franja": np.array(m_franja, np.int16),
        "seq": np.array(m_seq, np.int32),
    }
    return np.concatenate(S, 0), np.concatenate(T, 0), meta, n_skip


# --------------------------------------------------------------------------- #
#  Métricas que respaldan lo que la proyección sugiere
# --------------------------------------------------------------------------- #
def unitario(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def solapamiento_knn(S, T, franja, k):
    """Fracción de los k vecinos del maestro que siguen siendo vecinos en el estudiante."""
    vals = []
    for p in np.unique(franja):
        m = franja == p
        if m.sum() <= k + 1:
            continue
        a, b = unitario(S[m].astype(np.float64)), unitario(T[m].astype(np.float64))
        sa, sb = a @ a.T, b @ b.T
        np.fill_diagonal(sa, -np.inf)
        np.fill_diagonal(sb, -np.inf)
        ka = np.argpartition(-sa, k, axis=1)[:, :k]
        kb = np.argpartition(-sb, k, axis=1)[:, :k]
        vals.append(np.mean([len(set(ra) & set(rb)) for ra, rb in zip(ka, kb)]) / k)
    return float(np.mean(vals)) if vals else float("nan")


def rango_recuperacion(S, T, franja):
    """Posición del token correcto al buscar en el maestro con el embedding del estudiante."""
    rangos, candidatos = [], []
    for p in np.unique(franja):
        m = franja == p
        if m.sum() < 2:
            continue
        a, b = unitario(S[m].astype(np.float64)), unitario(T[m].astype(np.float64))
        sim = a @ b.T
        diag = np.diag(sim).copy()
        rangos.append((sim > diag[:, None]).sum(1) + 1)
        candidatos.append(m.sum())
    if not rangos:
        return None
    return np.concatenate(rangos), int(np.mean(candidatos))


def coseno_medio(S, T):
    return float(np.mean(np.sum(unitario(S.astype(np.float64)) * unitario(T.astype(np.float64)), 1)))


# --------------------------------------------------------------------------- #
#  Proyección
# --------------------------------------------------------------------------- #
def proyectar(X, metodo, n_neighbors, min_dist, seed, verbose):
    if metodo == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    try:
        import umap
    except ImportError:
        sys.exit("Falta umap-learn. Instálalo con:  pip install umap-learn\n"
                 "O usa --metodo pca (proyección lineal, mucho menos informativa).")
    red = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                    metric="cosine", random_state=seed, verbose=verbose)
    return red.fit_transform(X)


# --------------------------------------------------------------------------- #
#  Figuras
# --------------------------------------------------------------------------- #
def limpiar(ax, titulo):
    ax.set_title(titulo, fontsize=11, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for s in ax.spines.values():
        s.set_edgecolor("#CCCCCC")


def nube(ax, xy, color, s=2.5, alpha=0.5, **kw):
    return ax.scatter(xy[:, 0], xy[:, 1], c=color, s=s, alpha=alpha,
                      linewidths=0, rasterized=True, **kw)


def nube_fuentes(ax, xy_t, xy_s, s, alpha, seed=0):
    """Las dos fuentes en orden entremezclado: dibujadas una tras otra, la segunda tapa a la primera."""
    xy = np.concatenate([xy_t, xy_s], 0)
    col = np.array([C_MAESTRO] * len(xy_t) + [C_ESTUDIANTE] * len(xy_s))
    o = np.random.default_rng(seed).permutation(len(xy))
    nube(ax, xy[o], col[o], s=s, alpha=alpha)


def leyenda_fuentes(ax, etiq_maestro="maestro (VHuman3D)"):
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_MAESTRO, label=etiq_maestro),
                       Line2D([], [], marker="o", ls="", color=C_ESTUDIANTE, label="estudiante")],
              loc="best", fontsize=9, framealpha=0.9)


def fig_global(xy_t, xy_s, franja, salida, cab):
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))

    xy = np.concatenate([xy_t, xy_s], 0)
    fr = np.concatenate([franja, franja], 0)
    orden = np.random.default_rng(0).permutation(len(xy))     # evita que una fuente tape a la otra
    sc = nube(ax[0], xy[orden], fr[orden], cmap="viridis", vmin=0, vmax=PARTS - 1)
    limpiar(ax[0], "Coloreado por franja (0 = cabeza)")
    cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.02)
    cb.set_label("franja", fontsize=9)
    cb.set_ticks([0, 5, 10, 15])

    nube_fuentes(ax[1], xy_t, xy_s, s=2.5, alpha=0.45)
    limpiar(ax[1], "Coloreado por fuente")
    leyenda_fuentes(ax[1])

    fig.suptitle(cab, fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(salida, dpi=160)
    plt.close(fig)


def fig_franja(xy_t, xy_s, meta, p, salida, cab):
    m = meta["franja"] == p
    t, s = xy_t[m], xy_s[m]
    vistas = meta["vista"][m].astype(int)
    conds = np.array([c[:2] for c in meta["cond"][m]])

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.8))

    nube_fuentes(ax[0], t, s, s=11, alpha=0.6)
    limpiar(ax[0], "Fuente")
    leyenda_fuentes(ax[0], "maestro")

    # turbo y no twilight: twilight es casi blanco en los dos extremos y los 0 y 180 grados
    # desaparecerian sobre el fondo.
    sc = nube(ax[1], s, vistas, s=11, alpha=0.75, cmap="turbo", vmin=0, vmax=180)
    limpiar(ax[1], "Estudiante, por angulo de camara")
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.02)
    cb.set_label("grados", fontsize=9)

    for c in ("nm", "bg", "cl"):
        sel = conds == c
        if sel.any():
            nube(ax[2], s[sel], C_COND[c], s=11, alpha=0.75)
    limpiar(ax[2], "Estudiante, por condicion")
    ax[2].legend(handles=[Line2D([], [], marker="o", ls="", color=C_COND[c], label=NOMBRE_COND[c])
                          for c in ("nm", "bg", "cl") if (conds == c).any()],
                 loc="best", fontsize=9, framealpha=0.9)

    fig.suptitle(f"{cab}   |   franja {p} (filas {4 * p}-{4 * p + 3} de la silueta)",
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(salida, dpi=160)
    plt.close(fig)


def fig_pares(xy_t, xy_s, franja, salida, cab, n_pares, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xy_t))[:min(n_pares, len(xy_t))]
    fig, ax = plt.subplots(figsize=(7.6, 7.0))

    seg = np.stack([xy_t[idx], xy_s[idx]], 1)
    ax.add_collection(LineCollection(seg, colors="#999999", linewidths=0.6, alpha=0.55))
    nube_fuentes(ax, xy_t[idx], xy_s[idx], s=13, alpha=0.85)
    ax.autoscale_view()

    d = np.linalg.norm(xy_s[idx] - xy_t[idx], axis=1)
    radio = np.percentile(np.linalg.norm(xy_t - xy_t.mean(0), axis=1), 90)
    limpiar(ax, f"Desplazamiento maestro -> estudiante ({len(idx)} tokens)\n"
                f"mediana {np.median(d):.2f} = {100 * np.median(d) / max(radio, 1e-9):.0f} % "
                f"del radio de la nube ({radio:.2f})")
    leyenda_fuentes(ax, "maestro")
    fig.suptitle(cab, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(salida, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cfg", default="configs/student_distill_casiab.yaml")
    p.add_argument("--opengait-root", default="~/OpenGait")
    p.add_argument("--partition", default="")
    p.add_argument("--split", default="TEST_SET", choices=["TEST_SET", "TRAIN_SET"])
    p.add_argument("--max-seqs", type=int, default=240,
                   help="Secuencias muestreadas al azar (0 = todas). 240x4x16 ~ 15k tokens por fuente")
    p.add_argument("--frames-seq", type=int, default=4, help="Frames equiespaciados por secuencia")
    p.add_argument("--conds", nargs="*", default=[], help="Filtrar por nm/bg/cl (vacío = todas)")
    p.add_argument("--angulos", nargs="*", default=[], help="Filtrar por ángulo, p.ej. 000 090 180")
    p.add_argument("--franjas", nargs="*", type=int, default=[1, 8, 12],
                   help="Franjas con figura propia (vacío = ninguna)")
    p.add_argument("--metodo", default="umap", choices=["umap", "pca"])
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--knn", type=int, default=10, help="k del solapamiento de vecindarios")
    p.add_argument("--n-pares", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="Traza interna de UMAP (muy verbosa)")
    p.add_argument("--batch-frames", type=int, default=128)
    p.add_argument("--salida", default="umap")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    data_cfg, model_cfg = cfg["data_cfg"], cfg["model_cfg"]
    og_root = Path(args.opengait_root).expanduser().resolve()

    net = construir_modelo(model_cfg, og_root, args.device)
    it = cargar_pesos(net, Path(args.checkpoint).expanduser(), args.device)

    from data.transform import get_transform
    trf = get_transform(cfg["evaluator_cfg"]["transform"])[0]

    part_path = Path(args.partition).expanduser() if args.partition else \
        (og_root / data_cfg["dataset_partition"]).resolve()
    pids = json.loads(part_path.read_text())[args.split]
    print(f"[split] {args.split}: {len(pids)} sujetos | partición {part_path}")

    sil_root, teacher_root = Path(data_cfg["dataset_root"]), Path(data_cfg["teacher_root"])
    stats = data_cfg.get("teacher_stats")
    if stats:
        st = np.load(stats)
        t_mean = torch.as_tensor(st["mean"], dtype=torch.float32, device=args.device)
        t_std = torch.as_tensor(np.maximum(st["std"], 1e-6), dtype=torch.float32, device=args.device)
    else:
        t_mean = t_std = None

    seqs = muestrear(sil_root, teacher_root, pids, args.max_seqs,
                     [c[:2] for c in args.conds], args.angulos, args.seed)
    print(f"[muestra] {len(seqs)} secuencias x {args.frames_seq} frames x {PARTS} franjas")

    S, T, meta, n_skip = recolectar(net, trf, seqs, t_mean, t_std,
                                    args.frames_seq, args.device, args.batch_frames)
    print(f"[tokens] {len(S)} por fuente ({n_skip} secuencias descartadas por desalineación)")

    cos = coseno_medio(S, T)
    ov = solapamiento_knn(S, T, meta["franja"], args.knn)
    rec = rango_recuperacion(S, T, meta["franja"])

    print(f"\n=== Estructura del espacio (iteración {it}, muestra) ===")
    print(f"cos_sim de la muestra   : {cos:.4f}")
    print(f"solapamiento {args.knn}-NN        : {100 * ov:.1f} %  "
          f"(de los {args.knn} vecinos en el maestro, cuántos lo siguen siendo en el estudiante)")
    if rec is not None:
        r, cand = rec
        print(f"recuperación del token  : mediana rango {int(np.median(r))} de {cand} candidatos | "
              f"top-1 {100 * np.mean(r == 1):.1f} % | top-10 {100 * np.mean(r <= 10):.1f} % | "
              f"top-1% {100 * np.mean(r <= max(1, cand // 100)):.1f} %")

    print(f"\n[{args.metodo}] proyectando {2 * len(S)} puntos (puede tardar varios minutos)...")
    t0 = time.time()
    xy = proyectar(np.concatenate([T, S], 0), args.metodo,
                   args.n_neighbors, args.min_dist, args.seed, args.verbose)
    xy_t, xy_s = xy[:len(T)], xy[len(T):]
    print(f"[{args.metodo}] {time.time() - t0:.0f} s")

    out = Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)
    cab = (f"Iteración {it}  |  {args.split}  |  {len(seqs)} secuencias, {len(S)} tokens  |  "
           f"cos {cos:.3f}, solapamiento {args.knn}-NN {100 * ov:.0f} %")

    fig_global(xy_t, xy_s, meta["franja"], out / "umap_global.png", cab)
    print(f"[fig] {out / 'umap_global.png'}")
    for fp in args.franjas:
        if not (0 <= fp < PARTS):
            print(f"  [warn] franja {fp} fuera de rango, ignorada")
            continue
        dest = out / f"umap_franja_{fp:02d}.png"
        fig_franja(xy_t, xy_s, meta, fp, dest, cab)
        print(f"[fig] {dest}")
    fig_pares(xy_t, xy_s, meta["franja"], out / "umap_pares.png", cab, args.n_pares, args.seed)
    print(f"[fig] {out / 'umap_pares.png'}")

    npz = out / "umap_datos.npz"
    np.savez_compressed(npz, xy_maestro=xy_t, xy_estudiante=xy_s, iteracion=it,
                        cos=cos, solapamiento=ov, **meta)
    print(f"[datos] {npz}  (coordenadas y metadatos, para volver a graficar sin recalcular)")


if __name__ == "__main__":
    main()
