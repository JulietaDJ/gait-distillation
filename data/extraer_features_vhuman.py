import os
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import sys

sys.path.insert(0, "/opt3/data/julieta/vhuman3d/vhuman3d_deploy")

from vhuman3d.architectures import model_builder

def cargar_modelo(config_path, weights_path, device="cuda"):
    model = model_builder(config_path).to(device)
    model.load_state_dict(
        torch.load(weights_path, weights_only=True),
        strict=False,
    )
    model.eval()
    return model

def extraer_tensor_frame(model, rgb_np, device="cuda"):

    x = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).to(device)
    x = x.unsqueeze(0)

    with torch.no_grad():
        # 2. Backbone
        backbone_out = model.forward_backbone(x)
        
        # 3. Proyección y pos encoding
        backbone_out = model.backbone_projection(backbone_out)
        backbone_out = backbone_out + model.backbone_pos.unsqueeze(0).repeat(
            backbone_out.shape[0], 1, 1
        )
        
        # 4. Decoder 3D
        vol = model.embeddings_3d + model.pos_encoding
        vol = vol.unsqueeze(0).repeat(backbone_out.shape[0], 1, 1)
        for i in range(len(model.decoder_3d)):
            vol = model.decoder_3d[i](vol, backbone_out)
                    
        # 5. Reorganizar al volumen 3D
        vol = vol.view(vol.shape[0], *model.init_3d_grid_shape, model.embedding_dim_3d)
        
        vol_2d, _ = vol.max(dim=3) 

    return vol_2d.squeeze(0).cpu().numpy() 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", default="/opt3/data/julieta/CASIA-B-yolo/rgb/")
    parser.add_argument("--output", default="/opt3/data/julieta/vhuman_features")
    parser.add_argument("--config", default="vhuman3d/architectures/configs/vhuman3d_vitb.json")
    parser.add_argument("--weights", default="runs/run_1778669751/model_clean.pt")
    parser.add_argument("--max-sujetos", type=int, default=0)
    args = parser.parse_args()

    root_vhuman = Path("/opt3/data/julieta/vhuman3d/vhuman3d_deploy")
    os.chdir(root_vhuman)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando modelo en {device}...")
    model = cargar_modelo(args.config, args.weights, device)

    rgb_root = Path(args.rgb_root)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    sujetos = sorted([d for d in rgb_root.iterdir() if d.is_dir()])
    if args.max_sujetos > 0:
        sujetos = sujetos[:args.max_sujetos]

    for sujeto in sujetos:
        for cond in sorted(sujeto.iterdir()):
            if not cond.is_dir(): continue
            for ang in sorted(cond.iterdir()):
                if not ang.is_dir(): continue

                frames = sorted(ang.glob("*.png"))
                if not frames: continue

                feats_secuencia = []
                for frame_path in frames:
                    rgb_bgr = cv2.imread(str(frame_path))
                    if rgb_bgr is None: continue
                    rgb_np = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                    
                    try:
                        feat = extraer_tensor_frame(model, rgb_np, device) # [16, 16, 768]
                        feats_secuencia.append(feat)
                    except Exception as e:
                        print(f"  Error en {frame_path.name}: {e}")

                if feats_secuencia:
                    feats_array = np.stack(feats_secuencia, axis=0)
                    
                    out_dir = out_root / sujeto.name / cond.name / ang.name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / "features.npy", feats_array)
                    print(f"  {sujeto.name}/{cond.name}/{ang.name}: {feats_array.shape}")

if __name__ == "__main__":
    main()  

