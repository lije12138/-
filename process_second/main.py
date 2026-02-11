# -*- coding: utf-8 -*-
import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm

class TonguePipeline:
    def __init__(self, 
                 img_dir="datasets/images", 
                 mask_dir="datasets/images_mask", 
                 output_root="outputs_final",
                 calib_file="calibration_params.json"):
        
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_root)
        
        # 加载校准参数
        self.calib_params = self._load_calib(calib_file)
        
        # 创建输出目录结构
        self.dirs = {
            "clean": self.output_dir / "1_specularity_removed",
            "regions": self.output_dir / "2_subregions",
            "sep": self.output_dir / "3_coat_body",
            "report": self.output_dir / "4_reports"
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def _load_calib(self, path):
        """加载校准文件，如果不存在则使用默认值"""
        default = {"tx": 0, "ty": 0, "scale": 1.0, "rotation": 0}
        try:
            with open(path, 'r') as f:
                params = json.load(f)
                print(f"✅ 已加载校准参数: {params}")
                return params
        except FileNotFoundError:
            print("⚠️ 未找到校准文件，使用默认参数 (无变换)")
            return default

    def align_mask(self, mask_raw, img_shape):
        """
        核心步骤：基于重心的仿射变换
        """
        h, w = img_shape[:2]
        
        # 1. 计算 Mask 重心
        moments = cv2.moments(mask_raw)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            center = (cx, cy)
        else:
            center = (w // 2, h // 2)

        # 2. 构建变换矩阵 (以重心为锚点缩放)
        scale = self.calib_params.get("scale", 1.0)
        rotation = self.calib_params.get("rotation", 0)
        tx = self.calib_params.get("tx", 0)
        ty = self.calib_params.get("ty", 0)

        M = cv2.getRotationMatrix2D(center, rotation, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # 3. 应用变换 (Nearest保持二值边缘锐利)
        aligned_mask = cv2.warpAffine(mask_raw, M, (w, h), 
                                      flags=cv2.INTER_NEAREST, 
                                      borderValue=0)
        return aligned_mask

    def remove_specularity(self, img, mask):
        """去高光 (CLAHE + Inpaint)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 局部直方图均衡化增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(gray)
        
        # 提取高亮区域
        threshold = 230
        specular_mask = cv2.inRange(equalized, threshold, 255)
        specular_mask = cv2.bitwise_and(specular_mask, mask)
        
        # 膨胀处理
        kernel = np.ones((3,3), np.uint8)
        specular_mask = cv2.dilate(specular_mask, kernel, iterations=1)
        
        # 修复
        inpainted = cv2.inpaint(img, specular_mask, 3, cv2.INPAINT_TELEA)
        return inpainted

    def separate_coat_body(self, img, mask):
        """
        LAB + CLAHE + K-Means 分离舌苔/舌质
        """
        # 1. 阴影增强 (CLAHE on L-channel)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # 2. 补充 HSV 的 S 通道辅助
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        
        # 3. 提取特征 [a, b, s]
        mask_bool = (mask > 0)
        if np.sum(mask_bool) == 0:
             return mask, np.zeros_like(mask), img

        pixel_a = a[mask_bool].astype(float)
        pixel_b = b[mask_bool].astype(float)
        pixel_s = s[mask_bool].astype(float)
        
        # 归一化并加权
        # 舌质: 红(High a), 鲜艳(High s)
        # 舌苔: 白/黄(Low a, High b), 灰暗或极亮(Low s in shadows)
        features = np.column_stack((
            pixel_a * 2.0,  # 红绿色度 (关键)
            pixel_b * 0.8,  # 黄蓝色度
            pixel_s * 1.2   # 饱和度
        ))

        # 4. 聚类
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=3)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        
        # 5. 判别类别
        # 计算 "舌质得分" = a通道(红) + s通道(饱和)
        score0 = centers[0][0] + centers[0][2]
        score1 = centers[1][0] + centers[1][2]
        
        body_lbl = 0 if score0 > score1 else 1
        coat_lbl = 1 - body_lbl
        
        # 6. 生成 Mask
        h, w = mask.shape
        coat_mask = np.zeros((h, w), dtype=np.uint8)
        coat_mask[mask_bool] = (labels == coat_lbl).astype(np.uint8) * 255
        
        # 7. 形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        coat_mask = cv2.morphologyEx(coat_mask, cv2.MORPH_CLOSE, kernel, iterations=2) # 填空洞
        coat_mask = cv2.morphologyEx(coat_mask, cv2.MORPH_OPEN, kernel, iterations=1)  # 去噪点
        
        # 严格约束
        coat_mask = cv2.bitwise_and(coat_mask, mask)
        body_mask = cv2.bitwise_and(cv2.bitwise_not(coat_mask), mask)
        
        # 8. 可视化 (红=质, 绿=苔)
        vis = img.copy()
        vis[coat_mask > 0] = vis[coat_mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4
        vis[body_mask > 0] = vis[body_mask > 0] * 0.6 + np.array([0, 0, 255]) * 0.4
        
        return body_mask, coat_mask, vis


    def generate_regions(self, mask):
            """
            生成5分区，同时返回Mask字典和可视化图
            """
            h_img, w_img = mask.shape
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return {}, None  # 返回空字典
            
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # ---复现论文参数 (20%) ---
            tip_y = y + h * 0.80    # Bottom 1/5
            root_y = y + h * 0.20   # Top 1/5
            left_x = x + w * 0.20   # Left 1/5
            right_x = x + w * 0.80  # Right 1/5
            
            Y, X = np.indices((h_img, w_img))
            inside = (mask > 0)
            
            # 生成布尔Mask
            m_tip = inside & (Y >= tip_y)
            m_root = inside & (Y <= root_y) & ~m_tip
            m_left = inside & (X <= left_x) & ~m_tip & ~m_root
            m_right = inside & (X >= right_x) & ~m_tip & ~m_root
            m_center = inside & ~m_tip & ~m_root & ~m_left & ~m_right
            
            # 封装成字典返回
            sub_masks = {
                "Tip": m_tip,
                "Root": m_root,
                "Left": m_left,
                "Right": m_right,
                "Center": m_center
            }
            
            # 生成可视化图 (保持不变)
            vis = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            vis[m_tip] = [0, 0, 255]      # 红
            vis[m_root] = [255, 0, 0]     # 蓝
            vis[m_left] = [0, 255, 0]     # 绿
            vis[m_right] = [0, 255, 0]
            vis[m_center] = [0, 255, 255] # 黄
            
            return sub_masks, vis

    def create_report_image(self, original, clean, region_vis, sep_vis):
        """生成 2x2 的对比报告图"""
        def add_label(img, text):
            # 给图片加文字标签
            tmp = img.copy()
            cv2.putText(tmp, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            return tmp

        # 统一大小以便拼接
        h, w = original.shape[:2]
        target_size = (w, h)
        
        # 处理可能为空的 region_vis
        if region_vis is None: region_vis = np.zeros_like(original)

        row1 = np.hstack([add_label(original, "Original + Aligned Mask"), 
                          add_label(clean, "Specularity Removed")])
        row2 = np.hstack([add_label(region_vis, "Sub-regions"), 
                          add_label(sep_vis, "Body(Red) / Coat(Green)")])
        
        full_report = np.vstack([row1, row2])
        # 如果图太大，缩小一半保存
        if h > 1500:
            full_report = cv2.resize(full_report, (0,0), fx=0.5, fy=0.5)
            
        return full_report

    def run_batch(self, limit=None):
            """执行批处理 - 针对 dino_boxes 和 best_mask 格式优化"""
            
            # 1. 扫描所有的原图 (只寻找包含 _dino_boxes 的 jpg 文件)
            img_files = sorted(list(self.img_dir.glob("*_dino_boxes.jpg")))
            
            if not img_files:
                print(f"❌ 在 {self.img_dir} 中没找到 *_dino_boxes.jpg 文件，请检查路径。")
                return
    
            if limit:
                img_files = img_files[:limit]
                print(f"🔹 预分析模式：处理前 {limit} 张图像")
            else:
                print(f"🔹 全量模式：处理全部 {len(img_files)} 张图像")
    
            stats_data = []
    
            for img_path in tqdm(img_files):
                # --- 关键：提取 ID ---
                # 从 '0000_dino_boxes.jpg' 提取出 '0000'
                file_id = img_path.stem.split('_')[0]
                stem = file_id  # 后续保存文件统一用 ID 命名
                
                # --- 关键：匹配对应的 Mask ---
                # 寻找 '0000_best_mask.jpg' (或者 .png)
                mask_path = self.mask_dir / f"{file_id}_best_mask.jpg"
                if not mask_path.exists():
                    mask_path = self.mask_dir / f"{file_id}_best_mask.png"
                
                # 如果依然找不到，跳过
                if not mask_path.exists():
                    continue
    
                # --- A. 加载 ---
                # 使用 imdecode 兼容中文路径
                img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                mask_raw = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                
                if img is None or mask_raw is None: continue
                
                # 确保二值化
                _, mask_raw = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)
    
                # --- B. 对齐 Mask ---
                mask_aligned = cv2.resize(mask_raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_aligned = self.align_mask(mask_aligned, img.shape)
    
                # --- C. 去高光 ---
                img_clean = self.remove_specularity(img, mask_aligned)
                cv2.imwrite(str(self.dirs["clean"] / f"{stem}.jpg"), img_clean)
                
                # 准备灰度图用于统计
                img_gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    
                # --- D. 区域分割 ---
                sub_masks, region_vis = self.generate_regions(mask_aligned)
                if region_vis is not None:
                    cv2.imwrite(str(self.dirs["regions"] / f"{stem}_regions.png"), region_vis)
    
                # --- E. 苔质分离 ---
                body_mask, coat_mask, sep_vis = self.separate_coat_body(img_clean, mask_aligned)
                cv2.imwrite(str(self.dirs["sep"] / f"{stem}_vis.jpg"), sep_vis)
                cv2.imwrite(str(self.dirs["sep"] / f"{stem}_body.png"), body_mask)
                cv2.imwrite(str(self.dirs["sep"] / f"{stem}_coat.png"), coat_mask)
    
                # --- F. 生成报告 ---
                img_with_contour = img.copy()
                contours, _ = cv2.findContours(mask_aligned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_with_contour, contours, -1, (0, 0, 255), 3)
                
                report_img = self.create_report_image(img_with_contour, img_clean, region_vis, sep_vis)
                cv2.imwrite(str(self.dirs["report"] / f"{stem}_report.jpg"), report_img)
    
                # --- G. 收集数据 ---
                total_area = cv2.countNonZero(mask_aligned)
                coat_area = cv2.countNonZero(coat_mask)
                body_area = cv2.countNonZero(body_mask)
                
                row_data = {
                    "Filename": stem,
                    "Tongue_Area_Px": total_area,
                    "Coat_Area_Px": coat_area,
                    "Body_Area_Px": body_area,
                    "Coat_Ratio": round(coat_area / total_area, 4) if total_area else 0,
                    "Body_Ratio": round(body_area / total_area, 4) if total_area else 0,
                }
    
                # 灰度统计
                if sub_masks:
                    for name, m_bool in sub_masks.items():
                        pixels = img_gray[m_bool]
                        median_val = np.median(pixels) if len(pixels) > 0 else 0
                        row_data[f"Gray_{name}"] = median_val
                
                # 3. 统一添加一次
                stats_data.append(row_data)
                
            # 保存 CSV
            df = pd.DataFrame(stats_data)
            csv_path = self.output_dir / "processing_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✅ 处理完成！")
            print(f"   - 报告图片: {self.dirs['report']}")
            print(f"   - 数据表格: {csv_path}")
# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    # 实例化管道
    pipeline = TonguePipeline(
    #     img_dir="datasets/images", 
    #     mask_dir="datasets/images_mask",
    #     output_root="outputs_analysis_test",  # 结果保存在这里
    #     calib_file="calibration_params.json"
    # )
        img_dir="datasets/images_new", 
        mask_dir="datasets/images_mask_new",
        output_root="outputs_analysis_test3",  # 结果保存在这里
        calib_file="calibration_params.json"
    )
    
    # 设定 limit=5 进行预分析，或者设为 None 跑全量
    pipeline.run_batch(limit=None)

