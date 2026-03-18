"""
SAM 像素分割服务
接收图片和边界框，返回分割Mask
"""

import numpy as np
import torch
import cv2
import os
import sys
import gc
import base64
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import time
import logging
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sam_root = os.path.join(current_dir, "segment-anything")
sys.path.insert(0, sam_root)

from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

predictor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_sam_model():
    """加载SAM模型"""
    logger.info("加载SAM模型...")
    
    # SAM权重文件路径
    SAM_CKPT = ".\\segment-anything\\checkpoints\\sam_vit_h_4b8939.pth"
    
    if not os.path.isfile(SAM_CKPT):
        raise FileNotFoundError(f"SAM权重文件不存在：{SAM_CKPT}")
    
    sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
    sam_model = sam_model.to('cpu')  # 初始放在CPU
    sam_model.eval()
    
    predictor = SamPredictor(sam_model)
    
    logger.info("SAM模型加载完成")
    return predictor

def preprocess_image(img_bytes):
    """预处理图像：读取字节流，转换为RGB"""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return None
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    except Exception as e:
        logger.error(f"图像预处理失败: {e}")
        return None

def postprocess_mask(mask, image_source, kernel_size=3):
    """
    后处理Mask：形态学操作，保留最大连通域
    """
    if mask.dtype != np.uint8:
        binary_mask = (mask > 0).astype(np.uint8) * 255
    else:
        binary_mask = mask * 255
    
    # 调整Mask尺寸到原图大小
    target_height, target_width = image_source.shape[:2]
    if binary_mask.shape[0] != target_height or binary_mask.shape[1] != target_width:
        binary_mask_resized = cv2.resize(
            binary_mask,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        binary_mask_resized = binary_mask
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask_resized, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    blurred = cv2.GaussianBlur(cleaned, (5, 5), 0)
    _, smoothed = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    
    gray = cv2.cvtColor(image_source, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    refined = smoothed.copy()
    edge_pixels = (edges > 0)
    refined[edge_pixels] = binary_mask_resized[edge_pixels]
    
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
    
    # 调整回原图尺寸
    final_mask_resized = cv2.resize(final_mask, (image_source.shape[1], image_source.shape[0]))
    
    return (final_mask_resized > 0).astype(np.uint8)

def generate_point_prompts(bbox, H, W):
    """
    根据边界框生成SAM点提示
    """
    points = []
    labels = []
    
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    points.append([center_x, center_y])
    labels.append(1)
    
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
    
    return np.array(points), np.array(labels)

def mask_to_image(mask):
    """将Mask转换为图像字节流"""
    mask_img = (mask * 255).astype(np.uint8)
    is_success, buffer = cv2.imencode(".png", mask_img)
    if not is_success:
        return None
    return buffer.tobytes()

def mask_to_base64(mask):
    """将Mask转换为Base64编码"""
    mask_bytes = mask_to_image(mask)
    if mask_bytes is None:
        return None
    return base64.b64encode(mask_bytes).decode('utf-8')

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({"status": "healthy", "device": str(device)})

@app.route('/segment', methods=['POST'])
def segment():
    """
    分割接口
    接收：图片文件 + bbox参数
    返回：Mask图片
    """
    global predictor
    
    request_start = time.time()
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "没有上传图片文件"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "文件名为空"}), 400
        
        bbox_param = request.form.get('bbox')
        if not bbox_param:
            return jsonify({"error": "缺少bbox参数"}), 400
        
        try:
            bbox = json.loads(bbox_param)
            if len(bbox) != 4:
                return jsonify({"error": "bbox格式错误，应为[x1,y1,x2,y2]"}), 400
        except:
            return jsonify({"error": "bbox解析失败"}), 400
        
        img_bytes = file.read()
        
        image_source = preprocess_image(img_bytes)
        if image_source is None:
            return jsonify({"error": "图片格式错误"}), 400
        
        H, W = image_source.shape[:2]
        logger.info(f"处理图片: {file.filename}, 尺寸: {W}x{H}, bbox: {bbox}")
        
        if torch.cuda.is_available():
            predictor.model = predictor.model.to('cuda')
        
        predictor.set_image(image_source)
        
        input_point, input_label = generate_point_prompts(bbox, H, W)
        
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=np.array(bbox),
                multimask_output=True
            )
        
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        refined_mask = postprocess_mask(best_mask, image_source, kernel_size=3)
        
        predictor.reset_image()
        
        if torch.cuda.is_available():
            predictor.model = predictor.model.to('cpu')
        
        mask_area_ratio = refined_mask.sum() / (H * W)
        logger.info(f"Mask面积比例: {mask_area_ratio:.3f}, 置信度: {float(scores[best_mask_idx]):.3f}")
        
        mask_bytes = mask_to_image(refined_mask)
        if mask_bytes is None:
            return jsonify({"error": "Mask编码失败"}), 500
        
        process_time = time.time() - request_start
        
        return send_file(
            io.BytesIO(mask_bytes),
            mimetype='image/png',
            download_name=f'mask_{file.filename}.png'
        )
        
    except Exception as e:
        logger.error(f"分割失败: {e}", exc_info=True)
        
        if predictor is not None:
            predictor.reset_image()
            if torch.cuda.is_available():
                predictor.model = predictor.model.to('cpu')
                clear_memory()
        
        return jsonify({"error": str(e)}), 500

@app.route('/segment_json', methods=['POST'])
def segment_json():
    """
    分割接口（返回JSON）
    接收：图片文件 + bbox参数
    返回：包含Mask信息的JSON
    """
    global predictor
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "没有上传图片文件"}), 400
        
        file = request.files['image']
        bbox_param = request.form.get('bbox')
        
        if not bbox_param:
            return jsonify({"error": "缺少bbox参数"}), 400
        
        try:
            bbox = json.loads(bbox_param)
        except:
            return jsonify({"error": "bbox解析失败"}), 400
        
        img_bytes = file.read()
        
        image_source = preprocess_image(img_bytes)
        if image_source is None:
            return jsonify({"error": "图片格式错误"}), 400
        
        H, W = image_source.shape[:2]
        
        if torch.cuda.is_available():
            predictor.model = predictor.model.to('cuda')
        
        predictor.set_image(image_source)
        
        input_point, input_label = generate_point_prompts(bbox, H, W)

        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=np.array(bbox),
                multimask_output=True
            )
        
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        refined_mask = postprocess_mask(best_mask, image_source, kernel_size=3)
        
        predictor.reset_image()
        
        # 将模型移回CPU
        if torch.cuda.is_available():
            predictor.model = predictor.model.to('cpu')
            clear_memory()
        
        # 转换为Base64
        mask_base64 = mask_to_base64(refined_mask)
        
        contours, _ = cv2.findContours(
            (refined_mask * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        simplified_contours = []
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            simplified_contours.append(approx.reshape(-1, 2).tolist())
        
        return jsonify({
            "success": True,
            "mask_base64": mask_base64,
            "mask_area_ratio": float(refined_mask.sum() / (H * W)),
            "score": float(scores[best_mask_idx]),
            "contours": simplified_contours,
            "image_size": {"width": W, "height": H}
        })
        
    except Exception as e:
        logger.error(f"分割失败: {e}", exc_info=True)
        
        if predictor is not None:
            predictor.reset_image()
            if torch.cuda.is_available():
                predictor.model = predictor.model.to('cpu')
        
        return jsonify({"error": str(e)}), 500

@app.route('/segment_base64', methods=['POST'])


if __name__ == '__main__':
    logger.info("启动SAM服务...")
    
    predictor = load_sam_model()
    
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=False)