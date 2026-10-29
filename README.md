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

```bash
bash scripts/install_into_opengait.sh
python -m torch.distributed.run --nproc_per_node=1 scripts/train_distill.py \
    --cfgs configs/student_distill_casiab.yaml --phase train --log_to_file
```

Prueba rápida con 5 sujetos: `dataset_partition: <repo>/configs/CASIA-B-debug5.json`.

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
