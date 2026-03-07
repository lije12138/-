import os
import cv2
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# --- 环境与配置 ---
os.environ['OMP_NUM_THREADS'] = '1'
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
    "seed": 42
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(CONFIG["seed"])

# --- 数据处理类 (完全参照原代码) ---
class TongueHybridDataset:
    def __init__(self, label_csv, prior_csv, img_dir, mask_dir, config, is_train=True, scaler=None):
        self.config = config
        df_label = pd.read_csv(label_csv)
        def clean_id(x):
            s = str(x).split('-')[-1]
            return os.path.splitext(s)[0].strip()
        df_label['id_key'] = df_label['filename'].apply(clean_id)
        self.data_df = df_label.set_index('id_key')

        df_prior = pd.read_csv(prior_csv)
        first_col = df_prior.columns[0]
        df_prior[first_col] = df_prior[first_col].apply(lambda x: os.path.splitext(str(x))[0].strip())
        self.prior_df = df_prior.set_index(first_col).iloc[:, :349]

        self.samples = []
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_best_mask.jpg')]
        for m_file in mask_files:
            base_id = m_file.replace("_best_mask.jpg", "").strip()
            self.samples.append({'id': base_id})

        valid_ids = [s['id'] for s in self.samples]
        raw_priors = self.prior_df.loc[valid_ids].values.astype(np.float32)
        if is_train:
            self.scaler = StandardScaler()
            self.priors = self.scaler.fit_transform(raw_priors)
        else:
            self.scaler = scaler
            self.priors = self.scaler.transform(raw_priors)

# --- 训练逻辑 ---
train_ds = TongueHybridDataset(CONFIG['train_label'], CONFIG['train_prior'], CONFIG['train_img'], CONFIG['train_mask'], CONFIG, True)
test_ds = TongueHybridDataset(CONFIG['test_label'], CONFIG['test_prior'], CONFIG['test_img'], CONFIG['test_mask'], CONFIG, False, train_ds.scaler)

X_train = train_ds.priors
y_train = np.array([train_ds.data_df.loc[s['id'], CONFIG['target_labels']].values for s in train_ds.samples]).astype(int)
X_test = test_ds.priors
y_test = np.array([test_ds.data_df.loc[s['id'], CONFIG['target_labels']].values for s in test_ds.samples]).astype(int)

param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced_subsample'), param_grid, cv=3, n_jobs=-1, scoring='f1_macro')
rf.fit(X_train, y_train)

# 保存最优参数
with open("rf_best_params.txt", "w") as f:
    f.write(str(rf.best_params_))

# 评估
y_probs_list = rf.predict_proba(X_test)
y_probs = np.transpose([p[:, 1] for p in y_probs_list])
y_preds = (y_probs > 0.5).astype(int)

mAP = average_precision_score(y_test, y_probs, average='macro')
f1 = f1_score(y_test, y_preds, average='macro')
try:
    auc = roc_auc_score(y_test, y_probs, average='macro', multi_class='ovr')
except:
    auc = 0.0

# 写入统一评估文件
report_df = pd.DataFrame([{"Model": "Prior+RF", "mAP": mAP, "MacroF1": f1, "AUC": auc}])
report_df.to_csv("final_performance_report.csv", mode='a', header=not os.path.exists("final_performance_report.csv"), index=False)
print("RF Training Finished.")