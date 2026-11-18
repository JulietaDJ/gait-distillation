import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tordata

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D

import data.sampler as Samplers
from utils import get_attr_from, get_valid_args


class StudentDistill(BaseModel):
    """
    Estudiante para Feature Distillation basado en DeepGaitV2-2D.
    - Backbone 2D por frame + HPP de 16 franjas horizontales.
    - proj_head (MLP 2 capas) -> [N, T, 16, 768]; se descarta tras el entrenamiento.
    - Opcional (model_cfg.recognition_losses): ramas Triplet/CE como regularizador.

    Esta clase define su propio get_loader e inputs_pretreament, por lo que no
    hace falta parchear OpenGait (ni monkeypatching ni cambios en ts2var).
    """

    # ------------------------------------------------------------------ #
    #  Datos
    # ------------------------------------------------------------------ #
    def get_loader(self, data_cfg, train=True):
        from distillation_dataset import DistillationDataSet
        from distillation_collate import DistillationCollateFn

        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        dataset = DistillationDataSet(data_cfg, train)

        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        valid_args = get_valid_args(Sampler, sampler_cfg, free_keys=['sample_type', 'type'])
        sampler = Sampler(dataset, **valid_args)

        collate = DistillationCollateFn(
            dataset.label_set, sampler_cfg,
            teacher_stats=data_cfg.get('teacher_stats'),
            legacy_reduce_axis=data_cfg.get('legacy_reduce_axis', 0))

        num_workers = data_cfg['num_workers']
        return tordata.DataLoader(
            dataset=dataset, batch_sampler=sampler, collate_fn=collate,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None)

    def inputs_pretreament(self, inputs):
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch, teacher, mask = inputs
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(f"Número de entradas ({len(seqs_batch)}) != número de transforms ({len(seq_trfs)})")

        # Sin requires_grad en las entradas: no se necesita gradiente respecto a la silueta.
        seqs = [torch.from_numpy(np.asarray([trf(fra) for fra in seq], dtype=np.float32))
                .cuda(non_blocking=True) for trf, seq in zip(seq_trfs, seqs_batch)]
        labs = torch.as_tensor(labs_batch, dtype=torch.long).cuda(non_blocking=True)

        seqL = None
        if seqL_batch is not None:
            seqL = torch.from_numpy(seqL_batch).int().cuda(non_blocking=True)
            seqs = [s[:, :int(seqL.sum())] for s in seqs]

        if teacher is not None:
            teacher = teacher.cuda(non_blocking=True).float()
            mask = mask.cuda(non_blocking=True)
        return seqs, labs, typs_batch, vies_batch, seqL, teacher, mask

    # ------------------------------------------------------------------ #
    #  Red
    # ------------------------------------------------------------------ #
    def build_network(self, model_cfg):
        in_channels = model_cfg['Backbone']['in_channels']
        layers = model_cfg['Backbone']['layers']
        channels = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg.get('use_emb2', False)
        self.use_recognition = model_cfg.get('recognition_losses', False)
        self.parts = 16
        self.teacher_dim = model_cfg.get('teacher_dim', 768)
        proj_hidden = model_cfg.get('proj_hidden', 1024)

        strides = [[1, 1], [2, 2], [2, 2], [1, 1]]
        self.inplanes = channels[0]
        self.layer0 = nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True))
        self.layer1 = self.make_layer(BasicBlock2D, channels[0], strides[0], layers[0])
        self.layer2 = self.make_layer(BasicBlock2D, channels[1], strides[1], layers[1])
        self.layer3 = self.make_layer(BasicBlock2D, channels[2], strides[2], layers[2])
        self.layer4 = self.make_layer(BasicBlock2D, channels[3], strides[3], layers[3])

        self.HPP = HorizontalPoolingPyramid(bin_num=[self.parts])
        self.proj_head = nn.Sequential(
            nn.Linear(channels[3], proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, self.teacher_dim))

        # Ramas de reconocimiento (se mantienen para checkpoints/fase 2).
        self.FCs = SeparateFCs(self.parts, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(self.parts, channels[2],
                                       class_num=model_cfg['SeparateBNNecks']['class_num'])

    def make_layer(self, block, planes, stride, blocks_num):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))
        else:
            downsample = lambda x: x
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=[1, 1]))
        return nn.Sequential(*layers)

    def _recognition_head(self, out, N, T):
        out = out.view(N, T, *out.shape[1:])     # [N, T, C, H', W']
        out = out.max(dim=1)[0]                  # pooling temporal
        return self.FCs(self.HPP(out))           # [N, C_fc, 16]

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL, teacher_feat, mask = inputs
        sils = ipts[0]                           # [N, T, H, W]
        N, T, H, W = sils.shape

        out = sils.view(N * T, 1, H, W)
        out = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(out)))))  # [N*T, C4, H', W']

        if not self.training:
            feat_rec = self._recognition_head(out, N, T)
            embeddings = feat_rec if self.inference_use_emb2 else self.BNNecks(feat_rec)[0]
            return {'training_feat': {}, 'visual_summary': {},
                    'inference_feat': {'embeddings': embeddings}}

        feat_hpp = self.HPP(out).permute(0, 2, 1)            # [N*T, 16, C4]
        proj = self.proj_head(feat_hpp).view(N, T, self.parts, self.teacher_dim)

        training_feat = {
            'distill': {'student_feat': proj, 'teacher_feat': teacher_feat, 'mask': mask},
        }
        if self.use_recognition:
            embed_1 = self._recognition_head(out, N, T)
            _, logits = self.BNNecks(embed_1)
            training_feat['triplet'] = {'embeddings': embed_1, 'labels': labs}
            training_feat['softmax'] = {'logits': logits, 'labels': labs}

        return {'training_feat': training_feat, 'visual_summary': {}, 'inference_feat': {}}
