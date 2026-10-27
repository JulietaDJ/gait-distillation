import numpy as np
import torch
from opengait.data.collate_fn import CollateFn

class DistillationCollateFn(CollateFn):
    def __init__(self, label_set, sample_config):
        super().__init__(label_set, sample_config)

    def __call__(self, batch):
        original_batch = [item[:5] for item in batch]
        teacher_feats = [item[5] for item in batch]

        collated = super().__call__(original_batch)
        
        collated[0] = [[seq.astype(np.float32) for seq in feat] for feat in collated[0]]

        if self.sampler == 'fixed':
            frames_num = self.frames_num_fixed
        else:
            frames_num = teacher_feats[0].shape[0]

        teacher_batch = []
        for t_feat in teacher_feats:
            T = t_feat.shape[0]
            if T >= frames_num:
                indices = np.linspace(0, T - 1, frames_num).astype(int)
                t_sampled = t_feat[indices]
            else:
                pad = torch.zeros((frames_num - T, 16, 16, 768), dtype=t_feat.dtype)
                t_sampled = torch.cat([t_feat, pad], dim=0)
                
            # Colapsar la dimensión espacial
            t_sampled = t_sampled.mean(dim=1) # [frames_num, 16, 768]
            
            teacher_batch.append(t_sampled)

        teacher_batch = torch.stack(teacher_batch, dim=0) # [N, T, 16, 768]
        return collated[0], collated[1], collated[2], collated[3], collated[4], teacher_batch
