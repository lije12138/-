import os
import pandas as pd

def process_labels_to_csv(root_path, prefix):
    # 路径初始化
    images_dir = os.path.join(root_path, 'images')
    labels_dir = os.path.join(root_path, 'labels')
    classes_file = os.path.join(root_path, 'classes.txt')
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取类别名
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]

    data_list = []
    # 遍历图片文件夹获取文件名
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]

    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        label_file = os.path.join(labels_dir, f"{base_name}.txt")
        
        # 初始化当前图片的标签向量 (全0)
        row_data = {cls: 0 for cls in class_names}
        row_data['filename'] = f"{prefix}-{base_name}"
        
        # 如果存在对应的标签文件则读取
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        class_idx = int(parts[0])
                        if 0 <= class_idx < len(class_names):
                            row_data[class_names[class_idx]] = 1
        
        data_list.append(row_data)

    # 转换为DataFrame并调整列顺序
    df = pd.DataFrame(data_list)
    cols = ['filename'] + class_names
    df = df[cols]

    # 保存结果
    output_path = os.path.join(output_dir, f"{prefix}-labels.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"处理完成，结果已保存至: {output_path}")