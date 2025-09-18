import json
import os

# 你的 COCO 標註檔
coco_file = "annotations_fixed.json"   # 改成你的檔名
label_dir = "labels"             # 輸出 YOLO label 的資料夾

os.makedirs(label_dir, exist_ok=True)

with open(coco_file, "r", encoding="utf-8") as f:
    coco = json.load(f)

# 建立 image_id → file_name / size 的對應
id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

for ann in coco["annotations"]:
    image_id = ann["image_id"]
    file_name = id_to_filename[image_id]
    width, height = id_to_size[image_id]

    # 對應的 YOLO label 檔
    label_path = os.path.join(label_dir, file_name.rsplit(".", 1)[0] + ".txt")

    # segmentation 轉換成 YOLO 格式 (相對座標)
    for seg in ann["segmentation"]:
        norm_seg = []
        for i in range(0, len(seg), 2):
            x = seg[i] / width
            y = seg[i + 1] / height
            norm_seg.extend([x, y])

        # 類別 id 必須從 0 開始
        class_id = ann["category_id"] - 1

        with open(label_path, "a") as f:
            f.write(f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", norm_seg)) + "\n")
