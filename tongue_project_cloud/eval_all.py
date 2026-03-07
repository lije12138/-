import os
import cv2
import torch
import numpy as np
import pandas as pd
import timm
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score, roc_auc_score
from tqdm import tqdm

# --- 1. 基础配置 ---
BASE_DIR = "/root/autodl-tmp/tongue_project"
DATA_DIR = os.path.join(BASE_DIR, "data/upload")

CONFIG = {
    "train_img": os.path.join(DATA_DIR, "train/images"),
    "test_img": os.path.join(DATA_DIR, "test/images"),
    "train_mask": os.path.join(DATA_DIR, "images_train_v2_mask_filtered"),
    "test_mask": os.path.join(DATA_DIR, "images_test_v2_mask_filtered"),
    "train_label": os.path.join(DATA_DIR, "train-labels.csv"),
    "test_label": os.path.join(DATA_DIR, "test-labels.csv"),
    "train_prior": os.path.join(DATA_DIR, "train_features1.csv"),
    "test_prior": os.path.join(DATA_DIR, "test_features1.csv"),
    "target_labels": ['hongshe', 'zishe', 'pangdashe', 'shoushe', 'hongdianshe', 
                      'liewenshe', 'chihenshe', 'baitaishe', 'huangtaishe', 'huataishe'],
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "input_size": 384, "batch_size": 16
}

# --- 2. 核心算法：增加了阈值提取逻辑 ---
def get_academic_metrics(y_true, y_probs, model_name, method):
    """计算指标并返回结果字典及最优阈值列表"""
    res = {"Model": model_name, "Method": method}
    per_label_f1s = []
    per_label_aucs = []
    best_thresholds = [] # 新增：用于存储每个标签的最优阈值
    
    for i, label in enumerate(CONFIG['target_labels']):
        if len(np.unique(y_true[:, i])) < 2:
            res[f"{label}_F1"] = 0.0
            res[f"{label}_Pre"] = 0.0
            res[f"{label}_Rec"] = 0.0
            res[f"{label}_AUC"] = 0.5
            per_label_f1s.append(0.0)
            per_label_aucs.append(0.5)
            best_thresholds.append(0.5) # 默认值
            continue
            
        # 1. 计算 P-R 曲线寻找最优 F1 点及其对应的阈值
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        # 记录指标
        res[f"{label}_F1"] = round(f1_scores[best_idx], 4)
        res[f"{label}_Pre"] = round(precision[best_idx], 4)
        res[f"{label}_Rec"] = round(recall[best_idx], 4)
        
        # 记录该标签的最优阈值 (注意：thresholds 长度比 precision/recall 少 1)
        # best_idx 指向的是最优 F1，对应的阈值通常在 thresholds[best_idx]
        actual_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        best_thresholds.append(float(actual_threshold))
        
        # 2. 计算该类别的独立 AUC
        label_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        res[f"{label}_AUC"] = round(label_auc, 4)
        
        per_label_f1s.append(f1_scores[best_idx])
        per_label_aucs.append(label_auc)
    
    # 3. 汇总全域指标
    res["Macro_F1_Opt"] = round(np.mean(per_label_f1s), 4)
    res["Macro_AUC"] = round(np.mean(per_label_aucs), 4)
    res["mAP"] = round(average_precision_score(y_true, y_probs, average='macro'), 4)
        
    return res, best_thresholds

# --- 3. 图像预处理类 ---
class TongueEvalDataset(Dataset):
    def __init__(self, mode='original'):
        self.mode = mode
        df_label = pd.read_csv(CONFIG['test_label'])
        df_label['id_key'] = df_label['filename'].apply(lambda x: os.path.splitext(str(x).split('-')[-1])[0].strip())
        self.data_df = df_label.set_index('id_key')
        
        self.samples = []
        mask_dir = CONFIG['test_mask']
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_best_mask.jpg')]
        for m_file in mask_files:
            base_id = m_file.replace("_best_mask.jpg", "").strip()
            img_path = os.path.join(CONFIG['test_img'], base_id + ".jpg")
            if not os.path.exists(img_path): img_path = os.path.join(CONFIG['test_img'], base_id + ".JPG")
            if os.path.exists(img_path):
                self.samples.append({'id': base_id, 'img': img_path, 'mask': os.path.join(mask_dir, m_file)})
        
        self.transform = transforms.Compose([
            transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s['img'])
        mask = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        coords = np.column_stack(np.where(mask > 0))
        ymin, xmin = coords.min(axis=0); ymax, xmax = coords.max(axis=0)
        side = max(ymax - ymin, xmax - xmin)
        cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
        
        if self.mode == 'original':
            avg_col = img[mask > 0].mean(axis=0).tolist() if (mask > 0).any() else [127, 127, 127]
            img_canvas = np.full((side, side, 3), avg_col, dtype=np.uint8)
        else:
            img_canvas = np.zeros((side, side, 3), dtype=np.uint8)
            
        h, w = img.shape[:2]
        src_y1, src_y2 = max(0, cy-side//2), min(h, cy+side//2)
        src_x1, src_x2 = max(0, cx-side//2), min(w, cx+side//2)
        dst_y1, dst_x1 = src_y1 - (cy-side//2), src_x1 - (cx-side//2)
        img_canvas[dst_y1:dst_y1+(src_y2-src_y1), dst_x1:dst_x1+(src_x2-src_x1)] = img[src_y1:src_y2, src_x1:src_x2]
        
        mask_canvas = np.zeros((side, side), dtype=np.uint8)
        mask_canvas[dst_y1:dst_y1+(src_y2-src_y1), dst_x1:dst_x1+(src_x2-src_x1)] = mask[src_y1:src_y2, src_x1:src_x2]

        if self.mode == 'original':
            img_blur = cv2.GaussianBlur(img_canvas, (25, 25), 15)
            mask_soft = cv2.GaussianBlur(mask_canvas, (15, 15), 0) / 255.0
            img_f = (img_canvas * mask_soft[...,None] + img_blur * (1.0 - mask_soft[...,None])).astype(np.uint8)
        else:
            img_f = cv2.bitwise_and(img_canvas, img_canvas, mask=(mask_canvas > 0).astype(np.uint8))

        img_pil = Image.fromarray(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB))
        label = torch.tensor(self.data_df.loc[s['id'], CONFIG['target_labels']].values.astype(np.float32))
        return self.transform(img_pil), label

# --- 4. 运行评估流程 ---

academic_results = []
swin_original_thresholds = None # 用于单独保存

# (A) RandomForest
print(">>> [1/4] Evaluating Random Forest...")
# ... (RF 对齐逻辑保持不变)
train_df = pd.read_csv(CONFIG['train_prior'])
test_df = pd.read_csv(CONFIG['test_prior'])
train_labels_df = pd.read_csv(CONFIG['train_label'])
test_labels_df = pd.read_csv(CONFIG['test_label'])
for df in [train_df, test_df]: df.iloc[:,0] = df.iloc[:,0].apply(lambda x: os.path.splitext(str(x))[0].strip())
train_labels_df['id'] = train_labels_df['filename'].apply(lambda x: os.path.splitext(str(x).split('-')[-1])[0].strip())
test_labels_df['id'] = test_labels_df['filename'].apply(lambda x: os.path.splitext(str(x).split('-')[-1])[0].strip())
X_tr = train_df.set_index(train_df.columns[0]).iloc[:, :349]
y_tr = train_labels_df.set_index('id')[CONFIG['target_labels']]
X_ts = test_df.set_index(test_df.columns[0]).iloc[:, :349]
y_ts = test_labels_df.set_index('id')[CONFIG['target_labels']]
com_tr, com_ts = X_tr.index.intersection(y_tr.index), X_ts.index.intersection(y_ts.index)

rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(X_tr.loc[com_tr], y_tr.loc[com_tr])
rf_probs = np.transpose([p[:, 1] for p in rf.predict_proba(X_ts.loc[com_ts])])
res_rf, _ = get_academic_metrics(y_ts.loc[com_ts].values, rf_probs, "RandomForest", "Tabular")
academic_results.append(res_rf)

# (B) 深度学习模型
dl_tasks = [
    ("Swin_Original", "best_Swin-Base.pth", "original"),
    ("DenseNet_Original", "best_DenseNet201.pth", "original"),
    ("Swin_Black_BG", "best_Swin_Black_BG.pth", "black")
]

for name, weight, mode in dl_tasks:
    if not os.path.exists(weight): continue
    print(f">>> [Evaluating] {name}...")
    m_type = 'swin_base_patch4_window12_384' if 'swin' in name.lower() else 'densenet201'
    model = timm.create_model(m_type, pretrained=False, num_classes=10).to(CONFIG['device'])
    model.load_state_dict(torch.load(weight, map_location=CONFIG['device']))
    model.eval()
    
    loader = DataLoader(TongueEvalDataset(mode=mode), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            all_p.append(torch.sigmoid(model(imgs.to(CONFIG['device']))).cpu().numpy())
            all_l.append(labels.numpy())
    
    # 获取指标和阈值
    res_dl, thresholds = get_academic_metrics(np.vstack(all_l), np.vstack(all_p), name, mode)
    academic_results.append(res_dl)
    
    # 如果是 Swin_Original，记录阈值
    if name == "Swin_Original":
        swin_original_thresholds = dict(zip(CONFIG['target_labels'], thresholds))

# --- 5. 格式化并导出 ---
# 导出主报表
final_df = pd.DataFrame(academic_results)
ordered_cols = ['Model', 'Method']
for l in CONFIG['target_labels']:
    ordered_cols += [f"{l}_F1", f"{l}_Pre", f"{l}_Rec", f"{l}_AUC"]
ordered_cols += ['Macro_F1_Opt', 'Macro_AUC', 'mAP']
final_df[ordered_cols].to_csv("tongue_academic_report_full.csv", index=False)

# 导出 Swin_Original 专属阈值
if swin_original_thresholds:
    with open("swin_original_thresholds.json", "w", encoding='utf-8') as f:
        json.dump(swin_original_thresholds, f, indent=4, ensure_ascii=False)
    print("\n>>> Swin_Original Best Thresholds saved to: swin_original_thresholds.json")

print("\nSUCCESS: All files generated!")