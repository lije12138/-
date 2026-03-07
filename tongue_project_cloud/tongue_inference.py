import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import json
import timm
from PIL import Image
from torchvision import transforms

class TongueDiagnosisModel:
    def __init__(self, model_path, threshold_json_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 1. 定义标签映射（必须与评估脚本中的顺序完全一致）
        self.target_labels_en = [
            'hongshe', 'zishe', 'pangdashe', 'shoushe', 'hongdianshe', 
            'liewenshe', 'chihenshe', 'baitaishe', 'huangtaishe', 'huataishe'
        ]
        self.target_labels_cn = [
            '红色', '紫舌', '胖大舌', '瘦舌', '红点舌', 
            '裂纹舌', '齿痕舌', '白苔', '黄苔', '滑苔'
        ]

        # 2. 从 JSON 加载最优阈值
        self.thresholds = self._load_thresholds(threshold_json_path)
        
        # 3. 加载模型架构
        print(f"Loading Swin-Base model from {model_path}...")
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=10)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_thresholds(self, json_path):
        """解析 JSON 阈值文件并按列表顺序对齐"""
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Using default 0.5 for all labels.")
            return [0.5] * 10
        
        with open(json_path, 'r', encoding='utf-8') as f:
            thresh_dict = json.load(f)
        
        # 按英文标签顺序提取阈值，确保索引对齐
        ordered_thresh = [thresh_dict.get(label, 0.5) for label in self.target_labels_en]
        print(f"Successfully loaded thresholds for {len(ordered_thresh)} labels.")
        return ordered_thresh

    def _preprocess(self, img_path, mask_path):
        """完全复现 Swin-original 的高斯模糊预处理逻辑"""
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            raise ValueError(f"Could not read image or mask from provided paths.")

        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return np.zeros((384, 384, 3), dtype=np.uint8)
            
        ymin, xmin = coords.min(axis=0); ymax, xmax = coords.max(axis=0)
        side = max(ymax - ymin, xmax - xmin)
        cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
        
        avg_col = img[mask > 0].mean(axis=0).tolist()
        canvas = np.full((side, side, 3), avg_col, dtype=np.uint8)
        h, w = img.shape[:2]
        y1, y2, x1, x2 = max(0, cy-side//2), min(h, cy+side//2), max(0, cx-side//2), min(w, cx+side//2)
        dy1, dx1 = y1-(cy-side//2), x1-(cx-side//2)
        canvas[dy1:dy1+(y2-y1), dx1:dx1+(x2-x1)] = img[y1:y2, x1:x2]
        
        m_canvas = np.zeros((side, side), dtype=np.uint8)
        m_canvas[dy1:dy1+(y2-y1), dx1:dx1+(x2-x1)] = mask[y1:y2, x1:x2]
        
        img_blur = cv2.GaussianBlur(canvas, (25, 25), 15)
        mask_soft = cv2.GaussianBlur(m_canvas, (15, 15), 0) / 255.0
        img_f = (canvas * mask_soft[...,None] + img_blur * (1.0 - mask_soft[...,None])).astype(np.uint8)
        
        return cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)

    def predict(self, img_path, mask_path):
        """核心推理函数"""
        processed_img = self._preprocess(img_path, mask_path)
        input_tensor = self.transform(processed_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]
        
        # 组装 JSON 结果
        results = {}
        for i, label_cn in enumerate(self.target_labels_cn):
            # 将 numpy bool 转换为原生 int (1/0) 方便 JSON 序列化
            results[label_cn] = 1 if probs[i] >= self.thresholds[i] else 0
            
        return results

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 基础路径配置（请根据你的 Autodl 实际路径检查）
    BASE_DIR = "/root/autodl-tmp/tongue_project/data/upload"
    MODEL_FILE = "best_Swin-Base.pth"
    THRESH_FILE = "swin_original_thresholds.json"
    
    # 定义测试文件路径
    # 原图路径：test/images/A (186).jpg
    test_img_path = os.path.join(BASE_DIR, "test/images/A (186).jpg")
    # Mask路径：images_test_v2_mask_filtered/A (186)_best_mask.jpg
    test_mask_path = os.path.join(BASE_DIR, "images_test_v2_mask_filtered/A (186)_best_mask.jpg")

    # 如果是 .JPG 后缀，做一个兼容性处理
    if not os.path.exists(test_img_path):
        test_img_path = test_img_path.replace(".jpg", ".JPG")

    try:
        # 2. 初始化模型
        print(">>> Initializing Diagnosis System...")
        predictor = TongueDiagnosisModel(model_path=MODEL_FILE, threshold_json_path=THRESH_FILE)
        print("Model initialized successfully.\n")

        # 3. 运行单次预测
        print(f">>> Running prediction for: {os.path.basename(test_img_path)}")
        if not os.path.exists(test_img_path) or not os.path.exists(test_mask_path):
            print(f"Error: File not found!")
            print(f"Img exists: {os.path.exists(test_img_path)} ({test_img_path})")
            print(f"Mask exists: {os.path.exists(test_mask_path)} ({test_mask_path})")
        else:
            report = predictor.predict(test_img_path, test_mask_path)
            
            # 4. 输出预测结果 JSON
            print("\n" + "="*30)
            print("DIAGNOSIS REPORT (JSON):")
            print("="*30)
            print(json.dumps(report, ensure_ascii=False, indent=4))
            print("="*30)
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Run failed: {e}")
        import traceback
        traceback.print_exc()