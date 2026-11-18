"""
DataSet de destilación: devuelve (data_list, seq_info, teacher).

- data_list / seq_info: exactamente lo que devuelve el DataSet de OpenGait,
  para que el muestreo y las etiquetas sigan la lógica estándar.
- teacher: np.memmap con las features del maestro, [T, 16, 768] (formato
  compacto, float16) o [T, 16, 16, 768] (formato antiguo). Se abre con
  mmap_mode='r', así que el disco solo se lee al indexar los frames
  muestreados en el collate. None si falta o no coincide en longitud.

Claves nuevas en data_cfg:
    teacher_root: carpeta con <pid>/<cond>/<view>/features.npy
"""
from pathlib import Path

import numpy as np

try:  # mismo nombre de módulo que usa base_model.py de OpenGait
    from data.dataset import DataSet
except ImportError:  # pragma: no cover
    from opengait.data.dataset import DataSet

from utils import get_msg_mgr


class DistillationDataSet(DataSet):
    def __init__(self, data_cfg, training):
        super().__init__(data_cfg, training)
        if 'teacher_root' not in data_cfg:
            raise KeyError("data_cfg.teacher_root es obligatorio para la destilación")
        self.teacher_root = Path(data_cfg['teacher_root'])
        self.prefix = f"{self.dataset_name}-"
        self._warned = set()

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            get_msg_mgr().log_warning(msg)

    def _teacher_path(self, seq_info):
        pid = str(seq_info[0])
        if pid.startswith(self.prefix):  # en training OpenGait antepone "CASIA-B-"
            pid = pid[len(self.prefix):]
        return self.teacher_root / pid / seq_info[1] / seq_info[2] / "features.npy"

    def __getitem__(self, idx):
        data_list, seq_info = super().__getitem__(idx)
        n_frames = len(data_list[0])

        path = self._teacher_path(seq_info)
        if not path.exists():
            self._warn_once(str(path), f"[Distill] Falta maestro: {path} -> secuencia enmascarada")
            return data_list, seq_info, None

        teacher = np.load(path, mmap_mode='r')
        if teacher.shape[0] != n_frames:
            # Longitudes distintas = frames desalineados; no se rellena ni se recorta.
            self._warn_once(str(path),
                            f"[Distill] T maestro={teacher.shape[0]} != T siluetas={n_frames} "
                            f"en {path} -> secuencia enmascarada")
            return data_list, seq_info, None

        return data_list, seq_info, teacher
