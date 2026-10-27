import os
import numpy as np
import torch
import yaml
from pathlib import Path

from opengait.data.dataset import DataSet

class DistillationDataSet(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import sys
        cfgs_path = None
        if '--cfgs' in sys.argv:
            cfgs_path = sys.argv[sys.argv.index('--cfgs') + 1]

        if cfgs_path and Path(cfgs_path).exists():
            with open(cfgs_path, 'r') as f:
                cfgs = yaml.safe_load(f)
            self.teacher_root = Path(cfgs['data_cfg'].get('teacher_root', '/opt3/data/julieta/vhuman_features'))
        else:
            self.teacher_root = Path("/opt3/data/julieta/vhuman_features")

    def __getitem__(self, idx):
        # 1. Obtener info de la secuencia PRIMERO
        seq_info = self.seqs_info[idx]
        
        # 2. FILTRO DE PRUEBA (Solo procesar sujetos <= 5)
        try:
            pid_num = int(str(seq_info[0]).replace("CASIA-B-", ""))
        except ValueError:
            pid_num = 0
            
        if pid_num > 5:
            # Devolver un dummy válido para no romper el batch
            sils = np.zeros((30, 64, 44), dtype=np.float32)
            lab = 0
            typ, vie, seqL = seq_info[1], seq_info[2], 30
            teacher_tensor = torch.zeros((30, 16, 16, 768), dtype=torch.float32)
            return sils, lab, typ, vie, seqL, teacher_tensor

        # 3. Llamar al padre para obtener las siluetas
        original_data = super().__getitem__(idx)

        # 4. Desempaquetar
        if len(original_data) == 5:
            sils, lab, typ, vie, seqL = original_data
        elif len(original_data) == 2:
            sils, lab = original_data
            typ = seq_info[1]
            vie = seq_info[2]
            seqL = len(sils)
        else:
            raise ValueError(f"Formato no esperado: {len(original_data)}")

        # 5. Convertir siluetas a float32
        if isinstance(sils, list):
            sils = np.stack(sils, axis=0)
        sils = sils.astype(np.float32)

        # 6. Cargar el maestro desde el disco
        raw_pid = str(seq_info[0])
        pid = raw_pid.replace("CASIA-B-", "").zfill(3)
        cond = seq_info[1]
        view = seq_info[2]
        
        teacher_path = self.teacher_root / pid / cond / view / "features.npy"
        
        if teacher_path.exists():
            teacher_feat = np.load(teacher_path) # [T_total, 16, 16, 768]
            T = len(sils)
            
            if teacher_feat.shape[0] >= T:
                teacher_feat = teacher_feat[:T] 
            else:
                pad = np.zeros((T - teacher_feat.shape[0], 16, 16, 768), dtype=np.float32)
                teacher_feat = np.concatenate([teacher_feat, pad], axis=0)
                
            teacher_tensor = torch.from_numpy(teacher_feat).float()
        else:
            teacher_tensor = torch.zeros((len(sils), 16, 16, 768), dtype=torch.float32)

        return sils, lab, typ, vie, seqL, teacher_tensor
