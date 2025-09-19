import json
import glob
import os

def reorder_coco_categories(file_path, target_order):
    """
    重新排序COCO檔案中的category順序
    
    Args:
        file_path: COCO檔案路徑
        target_order: 目標順序列表，例如 ['left', 'right', 'straight']
    """
    
    # 讀取COCO檔案
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"處理檔案: {file_path}")
    
    # 顯示原始順序
    original_categories = sorted(data['categories'], key=lambda x: x['id'])
    print(f"原始順序: {[cat['name'] for cat in original_categories]}")
    
    # 建立名稱到原始資料的映射
    name_to_category = {cat['name']: cat for cat in data['categories']}
    
    # 建立舊ID到新ID的映射
    old_to_new_id = {}
    
    # 建立新的categories列表
    new_categories = []
    for new_id, category_name in enumerate(target_order, 1):
        if category_name in name_to_category:
            old_cat = name_to_category[category_name]
            old_to_new_id[old_cat['id']] = new_id
            
            # 建立新的category項目
            new_category = old_cat.copy()
            new_category['id'] = new_id
            new_categories.append(new_category)
        else:
            print(f"警告: 找不到類別 '{category_name}'")
    
    # 更新所有annotations中的category_id
    updated_count = 0
    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in old_to_new_id:
            ann['category_id'] = old_to_new_id[old_cat_id]
            updated_count += 1
    
    # 更新categories
    data['categories'] = new_categories
    
    # 顯示新順序
    print(f"新順序: {[cat['name'] for cat in new_categories]}")
    print(f"更新了 {updated_count} 個標註")
    
    # 儲存修正後的檔案
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"已儲存: {file_path}")
    print("-" * 50)

def batch_reorder_coco_files(pattern="*.json", target_order=['left', 'right', 'straight']):
    """
    批次處理多個COCO檔案
    
    Args:
        pattern: 檔案搜尋模式
        target_order: 目標順序
    """
    
    coco_files = glob.glob(pattern)
    
    if not coco_files:
        print(f"找不到符合 '{pattern}' 的檔案")
        return
    
    print(f"找到 {len(coco_files)} 個檔案")
    print(f"目標順序: {target_order}")
    print("=" * 50)
    
    for file_path in coco_files:
        try:
            reorder_coco_categories(file_path, target_order)
        except Exception as e:
            print(f"處理檔案 {file_path} 時發生錯誤: {e}")
            continue
    
    print("批次處理完成！")

# 使用範例
if __name__ == "__main__":
    # 方法1: 處理單一檔案
    reorder_coco_categories('result_coco_wei.json', ['left', 'right', 'straight'])
    
    # 方法2: 批次處理當前目錄下所有.json檔案
    #batch_reorder_coco_files("*.json", ['left', 'right', 'straight'])
    
    # 方法3: 指定特定檔案模式
    # batch_reorder_coco_files("coco_*.json", ['left', 'right', 'straight'])