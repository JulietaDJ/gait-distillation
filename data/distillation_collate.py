"""
Collate de destilación.

Los índices de frames se muestrean UNA sola vez por secuencia y se aplican
tanto a las siluetas como al maestro, de modo que el frame i del estudiante
corresponde al frame i del maestro.

Salida (7 elementos):
    [seqs, labs, typs, vies, seqL, teacher, mask]
    seqs, labs, typs, vies, seqL : idéntico al CollateFn de OpenGait
    teacher : torch.float16 [N, T, 16, 768]  (None si el sampler no es 'fixed')
    mask    : torch.bool    [N, T]           True = frame con maestro válido
"""
import math
import random

import numpy as np
import torch

try:
    from data.collate_fn import CollateFn
except ImportError:  # pragma: no cover
    from opengait.data.collate_fn import CollateFn


class DistillationCollateFn(CollateFn):
    def __init__(self, label_set, sample_config, teacher_stats=None,
                 legacy_reduce_axis=0, parts=16, dim=768):
        super().__init__(label_set, sample_config)
        self.parts, self.dim = parts, dim
        # Solo formato antiguo [T, 16, 16, 768]: eje de la rejilla (sin T) a promediar.
        # 0 reproduce el mean(dim=1) original.
        self.legacy_reduce_axis = legacy_reduce_axis
        self.mean = self.std = None
        if teacher_stats:
            st = np.load(teacher_stats)
            self.mean = st['mean'].astype(np.float32)
            self.std = np.maximum(st['std'].astype(np.float32), 1e-6)

    # ------------------------------------------------------------------ #
    def _sample_indices(self, seq_len):
        """Réplica de la lógica 'fixed' de CollateFn.sample_frames, devolviendo índices."""
        n = self.frames_num_fixed
        if self.allordered:
            return np.sort(np.random.choice(seq_len, n, replace=seq_len < n))
        if self.ordered:
            fs_n = n + self.frames_skip_num
            idx = list(range(seq_len))
            if seq_len < fs_n:
                it = math.ceil(fs_n / seq_len)
                idx = idx * it
            start = random.choice(range(0, len(idx) - fs_n + 1))
            window = np.sort(np.random.choice(fs_n, n, replace=False)) + start
            return np.asarray([idx[i] for i in window])
        return np.random.choice(seq_len, n, replace=seq_len < n)

    def _prepare_teacher(self, teacher, idx):
        t = np.asarray(teacher[np.sort(idx)], dtype=np.float32)  # lectura ordenada del memmap
        t = t[np.argsort(np.argsort(idx))]                         # restaurar el orden muestreado
        if t.ndim == 4:                                            # formato antiguo
            t = t.mean(axis=1 + self.legacy_reduce_axis)
        if self.mean is not None:
            t = (t - self.mean) / self.std
        return t

    # ------------------------------------------------------------------ #
    def __call__(self, batch):
        if self.sampler != 'fixed':
            # Evaluación ('all_ordered'): comportamiento estándar, sin maestro.
            out = super().__call__([(d, s) for d, s, _ in batch])
            return out + [None, None]

        feature_num = len(batch[0][0])
        n = self.frames_num_fixed
        fras = [[] for _ in range(feature_num)]
        labs, typs, vies, teachers, masks = [], [], [], [], []

        for data_list, seq_info, teacher in batch:
            idx = self._sample_indices(len(data_list[0]))
            for k in range(feature_num):
                fras[k].append(np.asarray(data_list[k][idx]))
            labs.append(self.label_set.index(seq_info[0]))
            typs.append(seq_info[1])
            vies.append(seq_info[2])

            if teacher is None:
                teachers.append(np.zeros((n, self.parts, self.dim), np.float32))
                masks.append(np.zeros(n, bool))
            else:
                teachers.append(self._prepare_teacher(teacher, idx))
                masks.append(np.ones(n, bool))

        teacher_b = torch.from_numpy(np.stack(teachers)).half()   # [N, T, 16, 768]
        mask_b = torch.from_numpy(np.stack(masks))                 # [N, T]
        return [fras, labs, typs, vies, None, teacher_b, mask_b]
