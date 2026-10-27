import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_cos=1.0, beta=1.0):
        super(DistillationLoss, self).__init__()
        self.w_l1 = w_l1
        self.w_cos = w_cos
        self.beta = beta
        # OpenGait necesita este atributo para ponderar la pérdida
        self.loss_term_weight = 1.0

    def forward(self, student_feat, teacher_feat):
        # --- PRINT DE DEPURACIÓN (Opcional, puedes quitarlo luego) ---
        # print(f"DEBUG Loss - student: {student_feat.shape}, teacher: {teacher_feat.shape}")
        
        # 1. Smooth L1 Loss (Huber Loss)
        # Es robusto a outliers, ideal para alinear la magnitud
        loss_l1 = F.smooth_l1_loss(student_feat, teacher_feat, beta=self.beta, reduction='mean')

        # 2. Cosine Distance Loss (Alineación Direccional)
        # Añadimos eps para evitar división por cero si un vector es todo ceros
        eps = 1e-8
        s_norm = student_feat / (student_feat.norm(p=2, dim=-1, keepdim=True) + eps)
        t_norm = teacher_feat / (teacher_feat.norm(p=2, dim=-1, keepdim=True) + eps)
        
        # Producto punto para similitud coseno
        cos_sim = (s_norm * t_norm).sum(dim=-1)
        
        # Clampeamos para asegurar que está en [-1, 1] y evitar NaNs numéricos
        cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
        
        # Distancia coseno = 1 - similitud
        loss_cos = 1.0 - cos_sim.mean()

        # 3. Pérdida total ponderada
        total_loss = self.w_l1 * loss_l1 + self.w_cos * loss_cos

        # 4. Diccionario de métricas para TensorBoard
        info = {
            'loss_l1': loss_l1.detach(),
            'loss_cos': loss_cos.detach()
        }
        
        # OpenGait espera una tupla (loss_total, info_dict)
        return total_loss, info
