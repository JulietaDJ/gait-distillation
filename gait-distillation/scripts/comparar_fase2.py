#!/usr/bin/env python3
"""
Reúne los JSON de evaluar_reconocimiento.py y produce la comparación de la fase 2.

Lee <metricas>/*.json, agrupa por etiqueta y escribe en --salida:
    tabla_fase2.md      Rank-1 por condición, Rank-5 y mAP medios, para los dos brazos en
                        su última iteración, la diferencia, y los puntos de partida
                        sin entrenar reconocimiento (backbone aleatorio y destilado)
    fase2_curvas.png    Rank-1 frente a iteración, un panel por condición
    fase2_angulos.png   Rank-1 por ángulo de la sonda en la última iteración
    resumen.json        todo lo anterior en números

Etiquetas que reconoce: scratch, distill (curvas) y cero_aleatorio, cero_destilado.

Uso:
    python scripts/comparar_fase2.py --metricas resultados_fase2/metricas --salida resultados_fase2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

COND = ("NM", "BG", "CL")
BRAZOS = {"scratch": "Desde cero", "distill": "Con backbone destilado"}
CERO = {"cero_aleatorio": "Backbone aleatorio, sin entrenar",
        "cero_destilado": "Backbone destilado, sin entrenar"}
# control en gris, tratamiento en el naranja del estudiante (mismo que en las figuras UMAP)
COLOR = {"scratch": "#6B6B6B", "distill": "#D1702F",
         "cero_aleatorio": "#6B6B6B", "cero_destilado": "#D1702F"}
MARCA = {"scratch": "o", "distill": "s", "cero_aleatorio": "o", "cero_destilado": "s"}
TINTA, TINTA_2, REJILLA = "#1F1F1F", "#5A5A5A", "#E4E4E1"


def cargar(dir_metricas):
    grupos = {}
    for f in sorted(Path(dir_metricas).glob("*.json")):
        r = json.loads(f.read_text())
        et = r.get("etiqueta") or f.stem.rsplit("-", 1)[0]
        grupos.setdefault(et, []).append(r)
    for et in grupos:
        grupos[et].sort(key=lambda r: r["iteracion"])
    return grupos


def fila(nombre, r):
    return (f"| {nombre} | " + " | ".join(f"{r[c]['R1']:.2f}" for c in COND)
            + f" | **{r['media']['R1']:.2f}** | {r['media']['R5']:.2f} | {r['media']['mAP']:.2f} |")


def fila_delta(a, b):
    d = lambda x, y: f"{y - x:+.2f}"
    return ("| Δ (destilado − cero) | " + " | ".join(d(a[c]["R1"], b[c]["R1"]) for c in COND)
            + f" | **{d(a['media']['R1'], b['media']['R1'])}** | {d(a['media']['R5'], b['media']['R5'])}"
            f" | {d(a['media']['mAP'], b['media']['mAP'])} |")


def primera_que_alcanza(curva, objetivo):
    for r in curva:
        if r["media"]["R1"] >= objetivo:
            return r["iteracion"]
    return None


def tabla(grupos):
    lineas = ["| Modelo | NM R1 | BG R1 | CL R1 | Media R1 | Media R5 | Media mAP |",
              "|---|---|---|---|---|---|---|"]
    for et, nom in CERO.items():
        if et in grupos:
            lineas.append(fila(nom, grupos[et][-1]))
    finales = {et: grupos[et][-1] for et in BRAZOS if et in grupos}
    for et, nom in BRAZOS.items():
        if et in finales:
            lineas.append(fila(f"{nom} ({finales[et]['iteracion'] // 1000}k it)", finales[et]))
    if len(finales) == 2:
        lineas.append(fila_delta(finales["scratch"], finales["distill"]))

    lineas += ["", "mAP por condición:", "",
               "| Modelo | NM | BG | CL |", "|---|---|---|---|"]
    for et, nom in {**CERO, **BRAZOS}.items():
        if et in grupos:
            r = grupos[et][-1]
            lineas.append(f"| {nom} | " + " | ".join(f"{r[c]['mAP']:.2f}" for c in COND) + " |")

    notas = []
    if len(finales) == 2:
        obj = finales["scratch"]["media"]["R1"]
        it = primera_que_alcanza(grupos["distill"], obj)
        notas.append(f"El brazo destilado alcanza el Rank-1 medio final del control ({obj:.2f} %) "
                     + (f"en la iteración {it}." if it is not None else "en ninguna iteración evaluada."))
        comunes = sorted({r["iteracion"] for r in grupos["scratch"]} & {r["iteracion"] for r in grupos["distill"]})
        if comunes:
            s = {r["iteracion"]: r["media"]["R1"] for r in grupos["scratch"]}
            d = {r["iteracion"]: r["media"]["R1"] for r in grupos["distill"]}
            notas.append("Δ Rank-1 medio a igual iteración: "
                         + ", ".join(f"{i // 1000}k {d[i] - s[i]:+.2f}" for i in comunes) + ".")
    if notas:
        lineas += [""] + notas
    return "\n".join(lineas) + "\n", finales


def estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_2, labelsize=9, length=0)
    ax.grid(axis="y", color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)


def leyenda(fig, claves, nombres):
    """Arriba, fuera de los paneles: dentro taparía datos en algún panel."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[k], marker=MARCA[k], lw=2, ms=7,
                mfc="white" if k.startswith("cero") else COLOR[k], ls="" if k.startswith("cero") else "-",
                label=nombres[k]) for k in claves]
    fig.legend(handles=h, loc="upper left", ncol=len(h), fontsize=8.5, frameon=False,
               labelcolor=TINTA_2, bbox_to_anchor=(0.003, 0.935), handlelength=2.2)


def etiquetas_finales(ax, puntos):
    """puntos: [(x, y)]. Escribe el valor junto a cada punto separando los que se pisan."""
    if not puntos:
        return
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    orden = sorted(range(len(puntos)), key=lambda i: puntos[i][1])
    pos = [puntos[i][1] for i in orden]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    dx = 0.025 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    for k, i in enumerate(orden):
        x, y = puntos[i]
        ax.text(x + dx, pos[k], f"{y:.1f}", va="center", fontsize=9, color=TINTA)


def fig_curvas(grupos, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brazos = [b for b in BRAZOS if b in grupos]
    ceros = [c for c in CERO if c in grupos]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    xmax = max(r["iteracion"] for b in brazos for r in grupos[b]) / 1000
    por_panel = []
    for ax, c in zip(axs, COND):
        estilo(ax)
        finales = []
        for b in brazos:
            x = [r["iteracion"] / 1000 for r in grupos[b]]
            y = [r[c]["R1"] for r in grupos[b]]
            ax.plot(x, y, color=COLOR[b], lw=2, marker=MARCA[b], ms=7, mec="white", mew=1.5, zorder=3)
            finales.append((x[-1], y[-1]))
        for k in ceros:                        # punto de partida: otro embedding, sin unir a la curva
            v = grupos[k][-1][c]["R1"]
            ax.plot([0], [v], ls="", marker=MARCA[k], ms=8, mfc="white", mec=COLOR[k], mew=2, zorder=3)
        ax.set_title(c, fontsize=11, color=TINTA, loc="left")
        ax.set_xlabel("iteración (miles)", fontsize=9, color=TINTA_2)
        ax.set_xlim(-xmax * 0.04, xmax * 1.14)
        por_panel.append(finales)
    for ax, finales in zip(axs, por_panel):     # tras fijar los límites y compartidos
        etiquetas_finales(ax, finales)
    axs[0].set_ylabel("Rank-1 (%)", fontsize=9, color=TINTA_2)
    leyenda(fig, brazos + ceros, {**BRAZOS, **CERO})
    fig.suptitle("Fase 2 · Rank-1 en TEST_SET de CASIA-B (excluye vista idéntica)",
                 fontsize=11.5, color=TINTA, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(destino, dpi=200, facecolor="white")
    plt.close(fig)


def fig_angulos(finales, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, c in zip(axs, COND):
        estilo(ax)
        for b, r in finales.items():
            v = [int(x) for x in r["vistas"]]
            ax.plot(v, r[c]["R1_por_vista"], color=COLOR[b], lw=2, marker=MARCA[b], ms=7,
                    mec="white", mew=1.5, zorder=3)
        ax.set_xticks(v)
        ax.set_xticklabels([f"{x}°" for x in v], fontsize=8)
        ax.set_title(c, fontsize=11, color=TINTA, loc="left")
        ax.set_xlabel("ángulo de la sonda", fontsize=9, color=TINTA_2)
    axs[0].set_ylabel("Rank-1 (%)", fontsize=9, color=TINTA_2)
    nombres = {b: f"{BRAZOS[b]} ({finales[b]['iteracion'] // 1000}k it)" for b in finales}
    leyenda(fig, list(finales), nombres)
    fig.suptitle("Fase 2 · Rank-1 por ángulo de la sonda, última iteración",
                 fontsize=11.5, color=TINTA, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(destino, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metricas", default="resultados_fase2/metricas")
    p.add_argument("--salida", default="resultados_fase2")
    args = p.parse_args()

    grupos = cargar(args.metricas)
    if not grupos:
        sys.exit(f"No hay JSON en {args.metricas}")
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    md, finales = tabla(grupos)
    (salida / "tabla_fase2.md").write_text(md)
    print(md)
    if any(b in grupos for b in BRAZOS):
        fig_curvas(grupos, salida / "fase2_curvas.png")
        fig_angulos(finales, salida / "fase2_angulos.png")
        print(f"[fig] {salida / 'fase2_curvas.png'}\n[fig] {salida / 'fase2_angulos.png'}")
    resumen = {et: [{k: r[k] for k in ("iteracion", "checkpoint", "embedding", *COND, "media")}
                    for r in rs] for et, rs in grupos.items()}
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=1))


if __name__ == "__main__":
    main()
