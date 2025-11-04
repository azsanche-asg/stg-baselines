import json
import os

import numpy as np
from tqdm import tqdm


def run_scenegraph(mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in tqdm(os.listdir(mask_dir)):
        if not fname.endswith("_mask.npy"):
            continue
        mask = np.load(os.path.join(mask_dir, fname))
        objs = np.unique(mask)[1:]
        rules = []
        for idx, obj_a in enumerate(objs):
            coords_a = np.argwhere(mask == obj_a)
            ya, xa = np.mean(coords_a, axis=0)
            for obj_b in objs[idx + 1 :]:
                coords_b = np.argwhere(mask == obj_b)
                yb, xb = np.mean(coords_b, axis=0)
                if xa < xb:
                    rules.append(f"{obj_a}_leftof_{obj_b}")
                if ya < yb:
                    rules.append(f"{obj_a}_above_{obj_b}")
        pred = {
            "rules": rules,
            "repeats": [],
            "depth": 1,
            "persist_ids": [],
            "motion": [],
        }
        out_path = os.path.join(out_dir, fname.replace("_mask.npy", "_pred.json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pred, f)
