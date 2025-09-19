import os
import shutil
import yaml
from pathlib import Path
import random
from sklearn.model_selection import KFold
import json

IMAGES_FOLDER = "C:/Users/ysann/Desktop/meizu/data/rawdata/images"  # 替換成你的圖片資料夾路徑
LABELS_FOLDER = "C:/Users/ysann/Desktop/meizu/data/rawdata/labels"  # 替換成你的標註檔案資料夾路徑
K_FOLDS = 5  # 五摺交叉驗證

def prepare_k_fold_dataset():
    """準備五摺交叉驗證資料集"""
    
    print("正在準備五摺交叉驗證資料集...")
    
    # 1. 創建基礎目錄結構
    base_dir = "dataset_k_fold"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # 為每一摺創建目錄
    for fold in range(K_FOLDS):
        directories = [
            f"{base_dir}/fold_{fold}/images/train",
            f"{base_dir}/fold_{fold}/images/val", 
            f"{base_dir}/fold_{fold}/labels/train",
            f"{base_dir}/fold_{fold}/labels/val"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    # 2. 獲取所有有效的圖片-標註對
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    images = []
    
    for filename in os.listdir(IMAGES_FOLDER):
        if Path(filename).suffix in image_exts:
            images.append(filename)
    
    # 3. 檢查對應的標註檔案
    available_pairs = []
    for img in images:
        label_name = Path(img).stem + '.txt'
        label_path = os.path.join(LABELS_FOLDER, label_name)
        if os.path.exists(label_path):
            available_pairs.append(img)
        else:
            print(f"找不到對應標註檔: {label_name}")
    
    if len(available_pairs) == 0:
        print("沒有找到任何有效的圖片-標註組合")
        return False
    
    print(f"找到 {len(available_pairs)} 個有效的圖片-標註對")
    
    # 4. 使用K-Fold分割資料
    random.seed(42)  # 固定隨機種子確保結果可重現
    random.shuffle(available_pairs)
    
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # 儲存摺疊資訊
    fold_info = {}
    
    # 5. 為每一摺分割和複製資料
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(available_pairs)):
        print(f"\n處理第 {fold_idx + 1} 摺...")
        
        # 獲取訓練和驗證資料
        train_images = [available_pairs[i] for i in train_indices]
        val_images = [available_pairs[i] for i in val_indices]
        
        print(f"  訓練資料: {len(train_images)} 張")
        print(f"  驗證資料: {len(val_images)} 張")
        
        # 儲存摺疊資訊
        fold_info[f'fold_{fold_idx}'] = {
            'train_images': train_images,
            'val_images': val_images,
            'train_count': len(train_images),
            'val_count': len(val_images)
        }
        
        # 複製訓練資料
        train_success = copy_files(train_images, fold_idx, 'train')
        
        # 複製驗證資料  
        val_success = copy_files(val_images, fold_idx, 'val')
        
        # 為每一摺創建 dataset.yaml
        create_fold_yaml(fold_idx)
        
        print(f"  成功複製: 訓練 {train_success}/{len(train_images)}, 驗證 {val_success}/{len(val_images)}")
    
    # 6. 儲存摺疊資訊到 JSON 檔案
    with open(f'{base_dir}/fold_info.json', 'w', encoding='utf-8') as f:
        json.dump(fold_info, f, indent=2, ensure_ascii=False)
    
    # 7. 創建總體配置檔案
    create_master_config(len(available_pairs))
    
    return True

def copy_files(image_list, fold_idx, subset_name):
    """複製圖片和標註檔案到指定摺疊的子集"""
    success_count = 0
    base_dir = "dataset_k_fold"
    
    for img in image_list:
        try:
            # 複製圖片
            src_img = os.path.join(IMAGES_FOLDER, img)
            dst_img = f"{base_dir}/fold_{fold_idx}/images/{subset_name}/{img}"
            shutil.copy2(src_img, dst_img)
            
            # 複製標註檔
            label_name = Path(img).stem + '.txt'
            src_label = os.path.join(LABELS_FOLDER, label_name)
            dst_label = f"{base_dir}/fold_{fold_idx}/labels/{subset_name}/{label_name}"
            shutil.copy2(src_label, dst_label)
            
            success_count += 1
            
        except Exception as e:
            print(f"  複製失敗 {img}: {e}")
    
    return success_count

def create_fold_yaml(fold_idx):
    """為每一摺創建 dataset.yaml 配置檔"""
    dataset_config = {
        'train': f'images/train',  # 相對於 dataset.yaml 的路徑
        'val': f'images/val',      # 相對於 dataset.yaml 的路徑
        'nc': 3,  # 類別數量
        'names': ['left','right','straight']  # 類別名稱
    }
    
    with open(f'dataset_k_fold/fold_{fold_idx}/dataset.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

def create_master_config(total_samples):
    """創建主配置檔案，包含所有摺疊的資訊"""
    master_config = {
        'project_info': {
            'name': 'Crosswalk Detection K-Fold Cross Validation',
            'total_samples': total_samples,
            'k_folds': K_FOLDS,
            'class_count': 1,
            'class_names': ['crosswalk']
        },
        'fold_configs': {}
    }
    
    for fold_idx in range(K_FOLDS):
        master_config['fold_configs'][f'fold_{fold_idx}'] = {
            'dataset_yaml': f'./dataset_k_fold/fold_{fold_idx}/dataset.yaml',
            'train_path': f'images/train',  # 相對路徑
            'val_path': f'images/val'       # 相對路徑
        }
    
    with open('dataset_k_fold/master_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(master_config, f, default_flow_style=False, allow_unicode=True)

def verify_k_fold_dataset():
    """驗證K摺資料集是否準備正確"""
    
    print(f"\n驗證五摺交叉驗證資料集...")
    base_dir = "dataset_k_fold"
    
    # 檢查基礎目錄
    if not os.path.exists(base_dir):
        print(f"缺少基礎目錄: {base_dir}")
        return False
    
    # 檢查每一摺
    total_train_files = 0
    total_val_files = 0
    
    for fold_idx in range(K_FOLDS):
        print(f"\n檢查第 {fold_idx + 1} 摺:")
        
        required_paths = [
            f"{base_dir}/fold_{fold_idx}/images/train",
            f"{base_dir}/fold_{fold_idx}/images/val",
            f"{base_dir}/fold_{fold_idx}/labels/train", 
            f"{base_dir}/fold_{fold_idx}/labels/val",
            f"{base_dir}/fold_{fold_idx}/dataset.yaml"
        ]
        
        fold_valid = True
        train_count = 0
        val_count = 0
        
        for path in required_paths:
            if not os.path.exists(path):
                print(f"  缺少: {path}")
                fold_valid = False
            else:
                if os.path.isdir(path):
                    file_count = len(os.listdir(path))
                    print(f"  {path}: {file_count} 個檔案")
                    if 'train' in path and 'images' in path:
                        train_count = file_count
                    elif 'val' in path and 'images' in path:
                        val_count = file_count
                else:
                    print(f"  {path}: 存在")
        
        if fold_valid:
            total_train_files += train_count
            total_val_files += val_count
            print(f"  第 {fold_idx + 1} 摺: 訓練 {train_count}, 驗證 {val_count}")
        else:
            return False
    
    # 檢查配置檔案
    config_files = [
        f"{base_dir}/fold_info.json",
        f"{base_dir}/master_config.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\n配置檔案存在: {config_file}")
        else:
            print(f"\n缺少配置檔案: {config_file}")
            return False
    
    print(f"\n總計: {total_train_files} 訓練檔案, {total_val_files} 驗證檔案")
    return True

def main():
    print("=" * 50)
    print(" Crosswalk Detection 五摺交叉驗證資料準備")
    print("=" * 50)
    print("此腳本準備五摺交叉驗證資料集，完成後請使用你的 train_code 開始訓練")
    print()
    
    # 顯示當前設定
    print(f"圖片資料夾: {IMAGES_FOLDER}")
    print(f"標註資料夾: {LABELS_FOLDER}")
    print(f"交叉驗證摺數: {K_FOLDS}")
    print()
    
    # 準備K摺資料集
    if prepare_k_fold_dataset():
        # 驗證結果
        if verify_k_fold_dataset():
            print(f"\n" + "=" * 50)
            print(f"五摺交叉驗證資料集準備完成！")
            print(f"=" * 50)
            print(f"\n資料集結構:")
            print(f"  dataset_k_fold/")
            print(f"  ├── fold_0/ (第1摺)")
            print(f"  ├── fold_1/ (第2摺)")
            print(f"  ├── fold_2/ (第3摺)")
            print(f"  ├── fold_3/ (第4摺)")
            print(f"  ├── fold_4/ (第5摺)")
            print(f"  ├── fold_info.json (摺疊詳細資訊)")
            print(f"  └── master_config.yaml (主配置檔)")
            print(f"\n每一摺包含:")
            print(f"  ├── images/train/ (訓練圖片)")
            print(f"  ├── images/val/   (驗證圖片)")
            print(f"  ├── labels/train/ (訓練標註)")
            print(f"  ├── labels/val/   (驗證標註)")
            print(f"  └── dataset.yaml  (該摺配置檔)")
            print(f"\n使用方式:")
            print(f"  在你的訓練代碼中循環使用每一摺:")
            print(f"  for fold in range(5):")
            print(f"      config_path = f'dataset_k_fold/fold_{{fold}}/dataset.yaml'")
            print(f"      # 使用該配置檔訓練模型")
        else:
            print(f"\n資料集驗證失敗，請檢查上述錯誤")
    else:
        print(f"\n資料準備失敗，請檢查錯誤訊息並修正")

if __name__ == "__main__":
    main()