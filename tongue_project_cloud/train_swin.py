import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import random

# --- 配置 ---
os.environ['OMP_NUM_THREADS'] = '1'
BASE_DIR = "/root/autodl-tmp/tongue_project"
DATA_DIR = os.path.join(BASE_DIR, "data/upload")
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")

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
    "input_size": 384, "batch_size": 16, "acc_steps": 1, "seed": 42
}

# --- 复用原代码工具函数 ---
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_v8_crop_params(mask_gray):
    coords = np.column_stack(np.where(mask_gray > 0))
    if coords.size == 0: return None
    ymin, xmin = coords.min(axis=0); ymax, xmax = coords.max(axis=0)
    side = max(ymax - ymin, xmax - xmin)
    cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
    return {'y1': cy - side // 2, 'y2': cy + side // 2, 'x1': cx - side // 2, 'x2': cx + side // 2, 'side': side}

def apply_v8_smart_crop(img, mask, params, avg_color):
    h, w = img.shape[:2]; side = params['side']
    img_canvas = np.full((side, side, 3), avg_color, dtype=np.uint8)
    mask_canvas = np.zeros((side, side), dtype=np.uint8)
    src_y1, src_y2 = max(0, params['y1']), min(h, params['y2'])
    src_x1, src_x2 = max(0, params['x1']), min(w, params['x2'])
    dst_y1, dst_x1 = src_y1 - params['y1'], src_x1 - params['x1']
    dst_y2, dst_x2 = dst_y1 + (src_y2 - src_y1), dst_x1 + (src_x2 - src_x1)
    if src_y2 > src_y1 and src_x2 > src_x1:
        img_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        mask_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return img_canvas, mask_canvas

class TongueHybridDataset(Dataset):
    def __init__(self, label_csv, prior_csv, img_dir, mask_dir, config, transform=None, is_train=True, scaler=None):
        self.config, self.img_dir, self.mask_dir, self.transform, self.is_train = config, img_dir, mask_dir, transform, is_train
        df_label = pd.read_csv(label_csv)
        df_label['id_key'] = df_label['filename'].apply(lambda x: os.path.splitext(str(x).split('-')[-1])[0].strip())
        self.data_df = df_label.set_index('id_key')
        df_prior = pd.read_csv(prior_csv)
        df_prior.iloc[:,0] = df_prior.iloc[:,0].apply(lambda x: os.path.splitext(str(x))[0].strip())
        self.prior_df = df_prior.set_index(df_prior.columns[0]).iloc[:, :349]
        self.samples = []
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_best_mask.jpg')]
        for m_file in mask_files:
            base_id = m_file.replace("_best_mask.jpg", "").strip()
            img_path = os.path.join(self.img_dir, base_id + ".jpg")
            if not os.path.exists(img_path): img_path = os.path.join(self.img_dir, base_id + ".JPG")
            self.samples.append({'id': base_id, 'img': img_path, 'mask': os.path.join(mask_dir, m_file)})
        valid_ids = [s['id'] for s in self.samples]
        raw_priors = self.prior_df.loc[valid_ids].values.astype(np.float32)
        if is_train:
            self.scaler = StandardScaler(); self.priors = self.scaler.fit_transform(raw_priors)
        else:
            self.scaler = scaler; self.priors = self.scaler.transform(raw_priors)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]; img = cv2.imread(s['img']); mask = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: return self.__getitem__((idx + 1) % len(self))
        if img.shape[:2] != mask.shape[:2]: mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        tongue_area = mask > 0
        avg_col = img[tongue_area].mean(axis=0).tolist() if tongue_area.any() else [127, 127, 127]
        params = get_v8_crop_params(mask)
        img_c, mask_c = apply_v8_smart_crop(img, mask, params, avg_col) if params else (img, mask)
        img_b = cv2.GaussianBlur(img_c, (25, 25), 15); mask_s = cv2.GaussianBlur(mask_c, (15, 15), 0) / 255.0
        img_f = (img_c * mask_s[...,None] + img_b * (1.0 - mask_s[...,None])).astype(np.uint8)
        img_pil = Image.fromarray(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB))
        return self.transform(img_pil), torch.tensor(self.priors[idx], dtype=torch.float32), torch.tensor(self.data_df.loc[s['id'], self.config['target_labels']].values.astype(np.float32))

# --- 训练循环 ---
seed_everything(CONFIG["seed"])
train_trans = transforms.Compose([transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
val_trans = transforms.Compose([transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_ds = TongueHybridDataset(CONFIG['train_label'], CONFIG['train_prior'], CONFIG['train_img'], CONFIG['train_mask'], CONFIG, train_trans, True)
test_ds = TongueHybridDataset(CONFIG['test_label'], CONFIG['test_prior'], CONFIG['test_img'], CONFIG['test_mask'], CONFIG, val_trans, False, train_ds.scaler)
train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)

model = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=len(CONFIG['target_labels']))
state_dict = torch.load(os.path.join(WEIGHT_DIR, "swin_base/pytorch_model.bin"), map_location='cpu')
for k in ["head.fc.weight", "head.fc.bias", "head.weight", "head.bias"]: 
    if k in state_dict: del state_dict[k]
model.load_state_dict(state_dict, strict=False)
model.to(CONFIG['device'])

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
criterion = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()
best_f1, best_metrics = 0, None
patience, EARLY_STOP = 0, 8
log_data = []

# 初始化时将 best_f1 改为 best_map
best_map = 0 
best_metrics = None
patience, EARLY_STOP = 0, 8
log_data = []

for epoch in range(30):
    model.train(); total_loss = 0; optimizer.zero_grad()
    for i, (imgs, priors, labels) in enumerate(tqdm(train_loader, desc=f"Swin Epoch {epoch}")):
        imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
        with torch.cuda.amp.autocast():
            loss = criterion(model(imgs), labels) / CONFIG['acc_steps']
        scaler.scale(loss).backward()
        if (i+1) % CONFIG['acc_steps'] == 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        total_loss += loss.item() * CONFIG['acc_steps']
    
    avg_loss = total_loss / len(train_loader)
    model.eval(); all_p, all_l = [], []
    with torch.no_grad():
        for imgs, _, labels in test_loader:
            outputs = model(imgs.to(CONFIG['device']))
            all_p.append(torch.sigmoid(outputs).cpu().numpy()); all_l.append(labels.numpy())
            
    all_p, all_l = np.vstack(all_p), np.vstack(all_l)
    mAP = average_precision_score(all_l, all_p, average='macro')
    f1 = f1_score(all_l, (all_p > 0.5).astype(int), average='macro')
    auc = roc_auc_score(all_l, all_p, average='macro', multi_class='ovr')
    
    log_data.append({"epoch": epoch, "loss": avg_loss, "mAP": mAP, "f1": f1, "auc": auc})
    pd.DataFrame(log_data).to_csv("training_log_swin.csv", index=False)
    
    # --- 核心修改：基于 mAP 进行评估和保存 ---
    if mAP > best_map:
        best_map = mAP
        best_metrics = {"Model": "Swin-Base", "mAP": mAP, "MacroF1": f1, "AUC": auc}
        torch.save(model.state_dict(), "best_Swin-Base.pth")
        patience = 0
        print(f" >>> New Best mAP: {best_map:.4f} (F1: {f1:.4f}) | Model Saved.")
    else:
        patience += 1
        print(f" >>> No improvement in mAP for {patience} epochs. Current Best: {best_map:.4f}")

    if patience >= EARLY_STOP: 
        print(f"Early stopping triggered at epoch {epoch}")
        break

pd.DataFrame([best_metrics]).to_csv("final_performance_report.csv", mode='a', header=not os.path.exists("final_performance_report.csv"), index=False)