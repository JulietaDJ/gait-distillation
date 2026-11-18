# gait-distillation

Feature distillation VHuman3D (RGB, congelado) → DeepGaitV2-2D (siluetas 64×44) sobre OpenGait.

## Pipeline

| Paso | Script | Salida |
|---|---|---|
| 1. Siluetas + RGB recortado | `data/procesar_casiab_yolo.py [--rgb-size H W]` | `CASIA-B-yolo/{silhouettes,rgb}/<pid>/<cond>/<view>/<frame>.png` |
| 2. Empaquetado OpenGait | `data/empaquetar_pkl.py` | `CASIA-B-yolo-pkl/<pid>/<cond>/<view>/<view>-sils.pkl` |
| 3. Diagnóstico del maestro | `data/extraer_features_vhuman.py --check` | claves sin cargar, forma de la rejilla, perfiles por eje |
| 4. Features del maestro | `data/extraer_features_vhuman.py` | `vhuman_features_v2/.../features.npy` fp16 `[T,16,768]` + `frames.txt` |
| 5. Estadísticas | `data/calcular_stats_maestro.py` | `teacher_stats.npz` (solo TRAIN_SET) |
| 6. Instalación en OpenGait | `scripts/install_into_opengait.sh` | enlaces en `opengait/modeling/{models,losses}` |
| 7. Entrenamiento | `scripts/train_distill.py` | `OpenGait/output/CASIA-B/StudentDistill/...` |
| 8. Evaluación + export | `data/evaluar_destilacion.py --checkpoint ... --export backbone_fase1` | `cos_sim`/CKA en TEST_SET, `backbone_fase1_{student,deepgaitv2}.pt` |
| 9. Visualización | `data/visualizar_silueta.py --checkpoint ...` | `visualizaciones/<pid>-<cond>-<view>{_tira.png,.gif}` |
| 10. Estructura del espacio | `data/umap_embeddings.py --checkpoint ...` | `umap/umap_{global,franja_NN,pares}.png` + `umap_datos.npz` |
| 11. Fase 2: A/B de reconocimiento | `scripts/run_fase2.py --backbone backbone_fase1_..._deepgaitv2.pt` | `resultados_fase2/{tabla_fase2.md,fase2_curvas.png,fase2_angulos.png,metricas/}` |

```bash
bash scripts/install_into_opengait.sh
python -m torch.distributed.run --nproc_per_node=1 scripts/train_distill.py \
    --cfgs configs/student_distill_casiab.yaml --phase train --log_to_file
```

Prueba rápida con 5 sujetos: `dataset_partition: <repo>/configs/CASIA-B-debug5.json`.

## Visualización por franja

El backbone baja la silueta de 64×44 a un mapa de 16×11 y la HPP toma una fila del mapa
por franja (`view(n, c, 16, -1)`), de modo que la franja *p* corresponde exactamente a las
filas `[4p, 4p+4)` de la silueta de entrada. `visualizar_silueta.py` se apoya en eso para
pintar el coseno de cada franja sobre el cuerpo, en lugar de en una gráfica aparte:

```bash
python data/visualizar_silueta.py \
    --checkpoint ~/OpenGait/output/CASIA-B/StudentDistill/StudentDistill_Phase1/checkpoints/StudentDistill_Phase1-10000.pt \
    --opengait-root ~/OpenGait --salida visualizaciones
```

Sin `--seqs` toma una secuencia por condición (`nm-05`, `bg-01`, `cl-01`) del primer sujeto
disponible del split, a 090°, y usa todos sus frames. Color viridis, claro = mejor;
`--metric l1` invierte la escala para que claro siga siendo mejor. La escala es fija
(0.40–0.90 en coseno) para que dos corridas sean comparables, y la referencia discontinua
marca el predictor trivial.

## Estructura del espacio aprendido (UMAP)

`umap_embeddings.py` ajusta un único UMAP sobre la unión de los embeddings del maestro y
del estudiante, de modo que ambos caen en el mismo sistema de coordenadas y la comparación
es directa: si el estudiante hubiera aprendido una representación desplazada respecto al
maestro — cosa que un coseno alto no descarta, porque es invariante a la norma — las dos
nubes saldrían separadas.

```bash
python data/umap_embeddings.py \
    --checkpoint ~/OpenGait/output/CASIA-B/StudentDistill/StudentDistill_Phase1/checkpoints/StudentDistill_Phase1-10000.pt \
    --opengait-root ~/OpenGait --salida umap
```

Por defecto muestrea 240 secuencias al azar del TEST_SET × 4 frames × 16 franjas (~15 k
tokens por fuente). Las figuras por franja aíslan una banda, lo que elimina la separación
trivial entre franjas y deja ver lo que varía dentro de ella (ángulo, condición).

UMAP puede inventar vecindarios que no existen en el espacio original, así que junto a las
figuras se informan dos cifras que hacen comprobable lo que la proyección sugiere: el
**solapamiento k-NN** (de los *k* vecinos de un token en el espacio del maestro, cuántos lo
siguen siendo en el del estudiante) y el **rango de recuperación** (buscando en el maestro
con el embedding del estudiante, en qué posición queda el token correcto). Requiere
`pip install umap-learn`; con `--metodo pca` cae a una proyección lineal, mucho menos
informativa pero sin dependencias nuevas.

## Lo que la fase 1 no demuestra

Reproducir las features del maestro no prueba que el estudiante infiera estructura 3D: el
coseno y el CKA miden parecido en el espacio del maestro, no reconstrucción. El decodificador
3D de VHuman3D (`upscaler_3d` + `out_3d`) no es reutilizable aquí, porque espera la rejilla
`[16,16,16,768]` completa y el estudiante predice la versión colapsada a `[16,768]`.

Por eso la fase 1 se valida aguas abajo, en la fase 2.

## Fase 2: reconocimiento de marcha

DeepGaitV2 en modo 2d entrenado dos veces sobre las mismas siluetas, con triplet y
cross-entropy: desde cero (`configs/fase2_scratch.yaml`) e inicializado con el backbone de
la fase 1 (`configs/fase2_distill.yaml`). Las dos configuraciones solo difieren en
`restore_hint` y `save_name`; `run_fase2.py` lo verifica antes de empezar y aborta si no.

```bash
python scripts/run_fase2.py --opengait-root ~/OpenGait --gpu 0 \
    --backbone ~/gait-distillation/backbone_fase1_20k_deepgaitv2.pt
```

Pasos (todos reanudables: lo ya evaluado no se repite salvo `--reevaluar`):

1. **Puntos de partida sin entrenar reconocimiento**: Rank-1/mAP con el embedding del
   backbone (HPP, sin cabeza) para un backbone aleatorio y para el destilado. Cuánta
   identidad trae ya la representación de la fase 1, sin haber visto una etiqueta.
2. **Entrenamiento** de cada brazo con OpenGait. En el destilado se exige que el log
   registre `Restore Parameters from`: sin eso entrenaría desde cero en silencio.
3. **Evaluación de cada checkpoint** (cada `save_iter`): permite ver si el destilado
   converge antes, además de dónde termina.
4. **Comparación**: `tabla_fase2.md`, `fase2_curvas.png` (Rank-1 frente a iteración) y
   `fase2_angulos.png` (Rank-1 por ángulo de la sonda).

`data/evaluar_reconocimiento.py` sustituye a la fase test de OpenGait, que exige
`evaluator_cfg.sampler.batch_size` igual al nº de GPUs y solo reporta Rank-1. Usa el
protocolo estándar de CASIA-B (galería nm-01..04; sondas NM nm-05/06, BG bg-01/02,
CL cl-01/02; se excluye la vista idéntica), reproduce exactamente el Rank-1 de OpenGait
y añade:

| Métrica | Qué mide |
|---|---|
| Rank-1 | fracción de sondas cuyo vecino más cercano en la galería es el sujeto correcto; la cifra de todos los papers |
| Rank-5 | el sujeto correcto está entre los 5 primeros; separa "no lo reconoce" de "lo tiene cerca" |
| mAP | precisión media sobre todo el ranking, contra la galería de todas las demás vistas |
| por ángulo | Rank-1 por vista de la sonda: dónde gana o pierde el backbone destilado |

La comparación es de una sola corrida por brazo (OpenGait fija la semilla), así que
diferencias de alrededor de un punto en Rank-1 están dentro del ruido y no deben leerse
como efecto del backbone.

## Cómo leer las métricas

`cos_sim` no se interpreta contra 0 sino contra el **baseline por franja**: la HPP da al
estudiante la identidad de la franja gratis, así que predecir la media de cada franja ya
alcanza ~0.41 en CASIA-B sin haber aprendido nada. `evaluar_destilacion.py` reporta ese
baseline sobre el propio split y el porcentaje de margen cerrado, que es la cifra
comparable entre corridas. `valid_ratio` debe ser 1.0: por debajo hay secuencias sin
maestro o con longitud distinta a las siluetas.

Las cifras del entrenamiento son sobre `TRAIN_SET` y no dicen si el mapeo generaliza a
identidades nuevas; para eso está el paso 8 sobre `TEST_SET`.

## Diseño

- `StudentDistill` sobrescribe `get_loader` e `inputs_pretreament`: no hay monkeypatching ni
  cambios en el núcleo de OpenGait (el parche de `ts2var` ya no es necesario para este modelo).
- El collate muestrea los índices de frames una vez y los aplica a siluetas y maestro.
- Secuencias sin maestro o con longitud distinta se enmascaran (`mask`) y no aportan pérdida.
- Sin augmentaciones geométricas: el maestro está precalculado.
- El maestro se lee con `mmap_mode='r'`: solo se leen del disco los frames muestreados.

## Maestro (VHuman3D): convenciones verificadas en su código

- Entrada NHWC, RGB en [0, 1]; `preprocessing` redimensiona a 518×518 y normaliza internamente.
- `preprocessing` no conserva proporción: el RGB se rellena a cuadrado (`--pad-factor`, 1.2 como el demo).
- Rejilla 16³: G0 horizontal, G1 vertical (0 = arriba), G2 profundidad (según `plot_3d` del demo).
  Max sobre G2 + mean sobre G0 → 16 franjas verticales, como la HPP del estudiante.
- `model_clean.pt` no incluye `backbone.*` (DINOv2 congelada desde torch hub): es esperado.
- `--check` mide qué franjas ocupa el cuerpo predicho para comprobar la alineación vertical.

Recomendado: regenerar el RGB con `procesar_casiab_yolo.py --rgb-size 256 256 --rgb-pad-factor 1.2`
(las siluetas no cambian) para no ampliar 64 px a 518 px.
