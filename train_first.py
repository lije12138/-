import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import gc
import psutil
import time
from transformers import AutoTokenizer
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_image, predict
from groundingdino.util import box_ops
from transformers import BertTokenizer, BertModel, BertConfig
current_dir = os.path.dirname(os.path.abspath(__file__))
sam_root = os.path.join(current_dir, "segment-anything")
sys.path.insert(0, sam_root)
from segment_anything import sam_model_registry, SamPredictor


BERT_LOCAL_PATH = "D:/vscode/cup/bert-base-uncased"
OUTPUT_DIR = "D:/vscode/cup/images_new"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"
# 手动构建tokenizer（绕过from_pretrained的路径检测）
tokenizer = BertTokenizer(
    vocab_file=os.path.join(BERT_LOCAL_PATH, "vocab.txt"), 
    do_lower_case=True,  
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    unk_token="[UNK]",
    mask_token="[MASK]"
)
config = BertConfig.from_json_file(os.path.join(BERT_LOCAL_PATH, "config.json"))
bert_model = BertModel.from_pretrained(
    pretrained_model_name_or_path=None,  
    config=config,
    state_dict=torch.load(os.path.join(BERT_LOCAL_PATH, "pytorch_model.bin"), map_location="cpu")
)
bert_model.eval()
def hijack_from_pretrained(*args, **kwargs):
    return tokenizer
AutoTokenizer.from_pretrained = hijack_from_pretrained
GROUNDINGDINO_ROOT = "D:\\vscode\\cup\\grounding-dino"  
sys.path.append(GROUNDINGDINO_ROOT)
sys.path.append("D:\\vscode\\cup\\grounding-dino")
from groundingdino.datasets.transforms import (
    RandomResize, 
    ToTensor, 
    Normalize,
    Compose
)


def clear_memory(verbose=False):
    """清理内存"""
    gc.collect()  # 垃圾回收
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理GPU缓存
        torch.cuda.synchronize()
    if verbose:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"内存使用情况: {memory_info.rss / 1024 / 1024:.2f} MB")
    return True
    print("内存已清理")


def check_memory_usage(threshold_mb=2048):
    # 检查内存使用是否超过阈值
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > threshold_mb:
            print(f"内存使用过高：{memory_mb:.2f} MB > {threshold_mb} MB")
            return False
        return True
    except:
        return True


def load_groundingdino_model_local(config_path, checkpoint_path):
    cfg = SLConfig.fromfile(config_path)
    cfg.text_encoder_type = "bert-base-uncased"
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.text_encoder = bert_model
    model.tokenizer = tokenizer
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    return model


def prepare_image_for_prediction(image_path, use_preprocess="none"):
    from PIL import Image
    
    grounding_dino_transform = Compose([
        RandomResize([800], max_size=1333),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])   
    try:
     
        image_source_pil = Image.open(image_path).convert("RGB")
        original_image_np = np.asarray(image_source_pil)
    
       
        if use_preprocess == "none":
            image_tensor, _ = grounding_dino_transform(image_source_pil, None)
            return original_image_np, image_tensor, original_image_np
    
       
        def apply_preprocessing(image_np, method):
            if method == "clahe":
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            elif method == "equalized":
                yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            elif method == "sharpened":
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
                return cv2.filter2D(image_np, -1, kernel)
            return image_np
    
     
        processed_np = apply_preprocessing(original_image_np, use_preprocess)
    
        
        processed_pil = Image.fromarray(processed_np.astype(np.uint8))
    
       
        processed_tensor, _ = grounding_dino_transform(processed_pil, None)
    
        return processed_np, processed_tensor, original_image_np
    except Exception as e:
        print(f"读取图像 {image_path} 失败: {e}")
        return None, None, None


def refine_mask(mask, image_source, kernel_size=3):
    """
    优化分割掩膜    
    Args:
        mask: 原始二值掩膜
        image_source: 原始图像（用于边缘检测）
        kernel_size: 形态学操作核大小
    """
 
    binary_mask = mask.astype(np.uint8) * 255
    
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)  
    
   
    blurred = cv2.GaussianBlur(cleaned, (5, 5), 0)
    _, smoothed = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    
   
    gray = cv2.cvtColor(image_source, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
   
    refined = smoothed.copy()
    edge_pixels = (edges > 0)
   
    refined[edge_pixels] = binary_mask[edge_pixels]
    

    dist_transform = cv2.distanceTransform(smoothed, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
 
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
     
        max_contour = max(contours, key=cv2.contourArea)
       
        final_mask = np.zeros_like(sure_fg)
        cv2.drawContours(final_mask, [max_contour], -1, 255, -1)
    else:
        final_mask = sure_fg
    
    return (final_mask > 0).astype(bool)


def run_tongue_segmentation(image_path, gd_model, sam_predictor, use_preprocess="clahe"):
    try:
        with torch.no_grad():
            image_source, image_tensor, _ = prepare_image_for_prediction(image_path, use_preprocess)
            if image_source is None:
                return None
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            else:
                image_tensor = image_tensor.cpu()
            boxes, logits, _ = predict(
                model=gd_model,
                image=image_tensor,
                caption="tongue body, not mouth, not lips, not teeth",
                box_threshold=0.35,
                text_threshold=0.25
            )
            del image_tensor
            if len(boxes) == 0:
                del boxes, logits
                return None
            H, W = image_source.shape[:2]
            boxes_xyxy =box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            if len(boxes_xyxy) == 0:
                del boxes, logits, boxes_xyxy
                return None
            best_box = boxes_xyxy[0].numpy().astype(int)
            del boxes, logits, boxes_xyxy
        # 在检测框中心添加一个正向点
            points, labels = [], []
            center_x = (best_box[0] + best_box[2]) // 2
            center_y = (best_box[1] + best_box[3]) // 2
            points.append([center_x, center_y])
            labels.append(1)
            x1, y1, x2, y2 = best_box
            for dx, dy in [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]:
                point_x = int(x1 + (x2 - x1) * dx)
                point_y = int(y1 + (y2 - y1) * dy)
                points.append([point_x, point_y])
                labels.append(1)  
            margin = 10
            points.append([x1 - margin, center_y])
            labels.append(0)
            points.append([x2 + margin, center_y])
            labels.append(0)
            input_point = np.array(points)
            input_label = np.array(labels)
            sam_predictor.set_image(image_source)
        with torch.no_grad():
            masks, scores, _ = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=best_box, 
                multimask_output=True
            )
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            refined_mask = refine_mask(best_mask, image_source, kernel_size=2)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(OUTPUT_DIR, f"{img_name}_mask.png")
        cv2.imwrite(mask_path, refined_mask.astype(np.uint8)*255)
        plt.close('all')
        sam_predictor.reset_image()
        result = {"image_path": image_path, "mask_path": mask_path, "score": float(scores[best_mask_idx])}
        del image_source, masks, scores, best_mask, refined_mask
        clear_memory()
        return result
    except Exception as e:
        print(f"图像分割失败 {image_path}: {e}")
        clear_memory()
        return None


class MemoryEfficientProcessor:
    # 内存高效处理器
    def __init__(self, gd_model, sam_predictor, output_dir, batch_size=20):
        self.gd_model = gd_model
        self.sam_predictor = sam_predictor
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.processed_count = 0
        self.failed_count = 0
    def process_batch(self, image_paths):
        results = []
        for i, img_path in enumerate(image_paths):
            try:
                if i % 5 == 0 and not check_memory_usage(3500):
                    print("内存过高，强制清理并等待...")
                    clear_memory(verbose=True)
                    time.sleep(2)
                result = run_tongue_segmentation(img_path, self.gd_model, self.sam_predictor, use_preprocess="clahe")
                if result:
                    results.append(result)
                    self.processed_count += 1
                    if self.processed_count % 50 == 0:
                        print(f"已处理 {self.processed_count} 张图像")
                else:
                    self.failed_count += 1
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                self.failed_count += 1
                clear_memory()
                time.sleep(1)
                continue
        return results
    def process_all_images(self, all_image_paths, max_images=None):
        if max_images:
            all_image_paths = all_image_paths[:max_images]
        total = len(all_image_paths)
        print(f"开始处理{total}张图片，批次大小：{self.batch_size}")
        for batch_idx in range(0, total, self.batch_size):
            batch_start = batch_idx
            batch_end = min(batch_idx + self.batch_size, total)
            batch_paths = all_image_paths[batch_start:batch_end]
            print(f"\n{'='*60}")
            print(f"处理批次 {batch_idx//self.batch_size + 1}/{(total-1)//self.batch_size + 1}")
            print(f"图片 {batch_start+1} 到 {batch_end}")
            print(f"{'='*60}")
            start_time = time.time()
            self.process_batch(batch_paths)
            print(f"批次完成，清理内存...")
            clear_memory(verbose=True)
            elapsed = time.time() - start_time
            remaining = (total - batch_end) * (elapsed / len(batch_paths))
            print(f"本批次耗时：{elapsed:.1f}秒，预计剩余时间：{remaining/60:.1f}分钟")
            time.sleep(1)
        print("全部处理成功！")


def main():
    GD_CONFIG = "D:\\vscode\\cup\\grounding-dino\\groundingdino\\config\\GroundingDINO_SwinT_OGC.py"
    GD_CKPT = "D:\\vscode\\cup\\dino\\groundingdino_swint_ogc.pth"
    SAM_CKPT = ".\\segment-anything\\checkpoints\\sam_vit_h_4b8939.pth"      
    for path, name in [(GD_CONFIG, "GD配置"), (GD_CKPT, "GD权重"), (SAM_CKPT, "SAM权重")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name}不存在：{path}")    
    print("加载Grounding DINO...")
    gd_model = load_groundingdino_model_local(GD_CONFIG, GD_CKPT)
    print("加载SAM...")
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU总内存: {total_memory:.2f} GB")
        if total_memory < 6:
            batch_size = 8
        elif total_memory < 8:
            batch_size = 12
        else:
            batch_size = 15
    sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).cuda() 
    sam_predictor = SamPredictor(sam_model)
    
    IMG_FOLDER = ".\\shezhenv3-coco\\images\\"
    img_paths = [os.path.join(IMG_FOLDER, f) for f in os.listdir(IMG_FOLDER) if f.lower().endswith(".jpg")]
    processor = MemoryEfficientProcessor(
        gd_model=gd_model,
        sam_predictor=sam_predictor,
        output_dir=OUTPUT_DIR,
        batch_size=batch_size
    )
    processor.process_all_images(img_paths, max_images=6000)
    del gd_model, sam_model, sam_predictor
    clear_memory(verbose=True)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，正在清理...")
        clear_memory(verbose=True)
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        clear_memory(verbose=True)
    print("\n🎉 全部处理完成！")