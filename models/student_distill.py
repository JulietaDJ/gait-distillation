import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D

class StudentDistill(BaseModel):
    """
    Estudiante para Feature Distillation basado en DeepGaitV2-2D.
    - Backbone: 2D (procesa frames individuales).
    - Extracción: Usa HPP para obtener 16 partes horizontales.
    - Cabeza de proyección: MLP 2 capas que mapea cada parte a 768 dims.
    """

    def build_network(self, model_cfg):
        mode = '2d'
        block = BasicBlock2D

        in_channels = model_cfg['Backbone']['in_channels']
        layers = model_cfg['Backbone']['layers']
        channels = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg.get('use_emb2', False)

        strides = [[1, 1], [2, 2], [2, 2], [1, 1]]

        self.inplanes = channels[0]
        # NOTA: Ya no usamos SetBlockWrapper
        self.layer0 = nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode)
        self.layer2 = self.make_layer(BasicBlock2D, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(BasicBlock2D, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(BasicBlock2D, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        # Cabezas de reconocimiento estándar
        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2],
                                       class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        # --- Cabeza de proyección (C_out -> 768) ---
        C_out = channels[3] 
        D_out = 768
        
        self.proj_head = nn.Sequential(
            nn.Linear(C_out, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, D_out), 
        )

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride), 
                nn.BatchNorm2d(planes * block.expansion)
            )
        else:
            downsample = lambda x: x
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=[1, 1]))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        #1. Desempaquetar (6 elementos)
        ipts, labs, typs, vies, seqL, teacher_feat = inputs
        
        # ipts[0]: [N, T, H, W]
        sils = ipts[0]
        N, T, H, W = sils.shape
        
        # Aplanar para convoluciones 2D
        sils = sils.view(N * T, 1, H, W) # [N*T, 1, H, W]
        
        # --- 1. Backbone ---
        out = self.layer0(sils)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out) # [N*T, C4, H', W']
        
        # --- 2. Extracción de Features para Destilación (HPP) ---
        hpp_out = self.HPP(out) # [N*T, C4, 16, 1]
        feat_hpp = hpp_out.squeeze(-1).permute(0, 2, 1).contiguous() # [N*T, 16, C4]
        
        # Cabeza de proyección
        proj = self.proj_head(feat_hpp) # [N*T, 16, 768]
        proj = proj.view(N, T, 16, 768) # [N, T, 16, 768]


        if self.training:
            # --- 3. Features de reconocimiento (para Triplet/Softmax) ---
            # Reconstruir la dimensión temporal para el TP
            out = out.view(N, T, *out.shape[1:]) # [N, T, C4, H', W']

            
            # TP espera [N, T, C, H, W] y hace max sobre T
            out_rec, _ = torch.max(out, dim=1) # [N, C4, H', W']
            
            out_rec = self.HPP(out_rec) # [N, C4, 16, 1]
            feat_rec = self.FCs(out_rec) # [N, C_fc, 16, 1]
            
            retval = {
                'training_feat': {
                    'distill': {
                        'student_feat': proj, # [N, T, 16, 768]
                        'teacher_feat': teacher_feat, # [N, T, 16, 768]
                    },
                    'triplet': {'embeddings': feat_rec, 'labels': labs},
                    'softmax': {'logits': feat_rec, 'labels': labs},
                },
                'visual_summary': {},
                'inference_feat': {}

            }
        else:
            # --- Inferencia: solo backbone ---
            out = out.view(N, T, *out.shape[1:]) # [N, T, C4, H', W']
            out_rec, _ = torch.max(out, dim=1)
            out_rec = self.HPP(out_rec)
            feat_rec = self.FCs(out_rec)
            embeddings = self.BNNecks(feat_rec)[0] if not self.inference_use_emb2 else feat_rec
            
            retval = {
                'training_feat': {},
                'inference_feat': {
                    'embeddings': embeddings,
                }
            }
        return retval
