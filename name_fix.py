import json

with open("result_coco_wei.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for img in data["images"]:
    # 找到第一個 "-" 後面的部分
    if "-" in img["file_name"]:
        img["file_name"] = img["file_name"].split("-", 1)[1]

with open("annotations_fixed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
