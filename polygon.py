import os, cv2, numpy as np
from glob import glob

IMG_DIR = "dataset/images"
MSK_DIR = "dataset/masks"
LBL_DIR = "dataset/labels"
CLASS_ID = 0  # crosswalk

os.makedirs(LBL_DIR, exist_ok=True)
img_paths = sorted(glob(os.path.join(IMG_DIR, "*.*")))

def mask_to_polys(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None: return []
    # ensure binary (255=fg, 0=bg)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    # get outer contours only
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < 200:   # skip tiny blobs
            continue
        # optional: simplify to reduce points
        eps = 0.003 * cv2.arcLength(c, True)  # tweak 0.002–0.01
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        polys.append(approx)
    return polys

for img_path in img_paths:
    name = os.path.splitext(os.path.basename(img_path))[0]
    # support .png or .jpg masks with same stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        mask_path = os.path.join(MSK_DIR, name + "_mask" + ext)
        if os.path.exists(mask_path): break
    else:
        print(f"[skip] no mask for {name}")
        continue

    img = cv2.imread(img_path)
    if img is None: 
        print(f"[skip] bad image {img_path}")
        continue
    H, W = img.shape[:2]

    polys = mask_to_polys(mask_path)
    if not polys:
        # no crosswalk → write empty label file
        open(os.path.join(LBL_DIR, name + ".txt"), "w").close()
        continue

    with open(os.path.join(LBL_DIR, name + ".txt"), "w") as f:
        for poly in polys:
            # normalize to [0,1]
            xs = (poly[:,0] / W).clip(0,1)
            ys = (poly[:,1] / H).clip(0,1)
            coords = np.stack([xs, ys], axis=1).reshape(-1)
            line = str(CLASS_ID) + " " + " ".join(f"{v:.6f}" for v in coords)
            f.write(line + "\n")
