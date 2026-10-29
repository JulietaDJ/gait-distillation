import torch
import torch.nn.functional as F

from .base import BaseLoss


class DistillationLoss(BaseLoss):
    """
    Smooth L1 (magnitud) + distancia coseno (dirección) entre estudiante y maestro.

    student_feat, teacher_feat : [N, T, P, D]
    mask                       : [N, T] bool, True = frame con maestro válido.
                                 Los frames enmascarados no aportan pérdida ni gradiente.
    teacher_norm               : 'none' | 'layernorm'. 'layernorm' normaliza el maestro por
                                 token (sin parámetros); útil si no se usan teacher_stats.
    """

    def __init__(self, w_l1=1.0, w_cos=1.0, beta=1.0, teacher_norm='none',
                 loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        if teacher_norm not in ('none', 'layernorm'):
            raise ValueError(f"teacher_norm no soportado: {teacher_norm}")
        self.w_l1, self.w_cos, self.beta = w_l1, w_cos, beta
        self.teacher_norm = teacher_norm

    def forward(self, student_feat, teacher_feat, mask=None):
        s = student_feat.float()                      # fuera de autocast: cálculo en fp32
        t = teacher_feat.float()
        if self.teacher_norm == 'layernorm':
            t = F.layer_norm(t, t.shape[-1:])

        if mask is None:
            mask = torch.ones(s.shape[:2], dtype=torch.bool, device=s.device)
        m = mask.to(s.device).float()                 # [N, T]
        n_valid = m.sum()

        l1 = F.smooth_l1_loss(s, t, beta=self.beta, reduction='none').mean(dim=(-1, -2))  # [N, T]
        cos = F.cosine_similarity(s, t, dim=-1, eps=1e-6).mean(dim=-1)                    # [N, T]

        if n_valid == 0:  # batch sin maestro: pérdida nula pero conectada al grafo
            zero = s.sum() * 0.0
            return zero, {'loss_l1': zero.detach(), 'loss_cos': zero.detach(),
                          'cos_sim': zero.detach(), 'valid_ratio': zero.detach()}

        loss_l1 = (l1 * m).sum() / n_valid
        cos_sim = (cos * m).sum() / n_valid
        loss_cos = 1.0 - cos_sim
        total = self.w_l1 * loss_l1 + self.w_cos * loss_cos

        info = {
            'loss_l1': loss_l1.detach(),
            'loss_cos': loss_cos.detach(),
            'cos_sim': cos_sim.detach(),
            'valid_ratio': (n_valid / m.numel()).detach(),
        }
        return total, info
