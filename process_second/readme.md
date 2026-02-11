# TongueExpert 图像预处理管道使用手册

## 1. 项目概述

**TonguePipeline** 是一个自动化的舌象预处理与特征提取工具。它能够批量处理原始舌象拍摄数据，执行以下核心任务：

1. **Mask 对齐**：基于重心和仿射变换参数，将分割 Mask 与原图精准对齐。

2. **去高光**：消除舌面反光，还原真实纹理。

3. **区域划分**：基于几何规则将舌面划分为舌尖、舌根、舌中及左右边缘。

4. **苔质分离**：利用 Lab+HSV 色彩聚类算法，自动分割舌苔与舌质。

5. **报表生成**：生成可视化对比图及包含定量指标（面积占比、区域灰度）的 CSV 数据表。

6. 统计学验证与分析。

---

## 2. 环境依赖 (Prerequisites)

请确保您的 Python 环境已安装以下库：

```Bash
pip install opencv-python numpy pandas scikit-learn tqdm
```

---

## 3. 数据准备 (Data Preparation)

代码对文件命名和目录结构有严格要求：

### 3.1 目录结构

建议的项目文件夹结构如下：

```Plaintext
Project_Root/
├── main.py                  # 主程序代码
├── calibration_params.json  # 校准参数文件
├── datasets/                # 数据根目录
│   ├── images_new/          # 存放原图
│   └── images_mask_new/     # 存放 Mask 图
└── outputs_analysis_test3/  # (自动生成) 结果输出目录
│   ├── 01_specularity_removed/# 步骤1: 去除高光/修复后的图像
│   ├── 02_subregions/         # 步骤2: 5分区可视化图 (Tip, Root, Center...)
│   ├── 03_tissue_seg/         # 步骤3: 苔质分离结果 (Masks & Visualization)
│   ├── 04_reports/            # [重点] 2x2 综合对比拼图 (用于报告插图)
│   └── 5_hypothesis_testing/ #  统计分析结果 (图表+CSV)
│   └── processing_summary.csv # [数据] 包含面积、占比等定量数据的表格
```

### 3.2 文件命名规范 (关键)

程序通过**文件名中的 ID** 来匹配原图和 Mask。

- **原图命名**：必须包含 `_dino_boxes` 后缀。
  
  - 示例：`0001_dino_boxes.jpg` (程序会提取 `0001` 作为 ID)

- **Mask 命名**：必须包含 `_best_mask` 后缀。
  
  - 示例：`0001_best_mask.jpg` 或 `0001_best_mask.png`

> **注意**：如果原图叫 `A_dino_boxes.jpg`，Mask 必须叫 `A_best_mask.png`。

---

## 4. 参数配置 (Configuration)

在项目根目录下创建一个 `calibration_params.json` 文件，用于控制 Mask 的几何变换参数。

**文件内容示例：**

JSON

```
{
    "scale": 1.0,
    "rotation": 0,
    "tx": 0,
    "ty": 0
}
```

- **scale**: 缩放比例 (1.0 为不缩放)。

- **rotation**: 旋转角度 (正数顺时针，负数逆时针)。

- **tx**: 水平平移像素 (正数向右)。

- **ty**: 垂直平移像素 (正数向下)。

---

## 5. 运行代码

### 5.1 修改路径配置

在代码底部的 `if __name__ == "__main__":` 区域修改输入输出路径：

```Python
if __name__ == "__main__":
    pipeline = TonguePipeline(
        img_dir="datasets/images_new",       # 原图文件夹路径
        mask_dir="datasets/images_mask_new", # Mask文件夹路径
        output_root="outputs_analysis_test3",# 结果输出路径
        calib_file="calibration_params.json" # 参数文件
    )

    # limit=None 表示处理文件夹内所有图片
    # limit=5 表示只处理前5张（用于测试）
    pipeline.run_batch(limit=None)
```

### 5.2 启动程序

在终端运行：

```Bash
python main.py
```

程序将显示进度条，并在控制台输出处理日志。

---

## 6. 输出结果解读 (Outputs)

程序运行结束后，`outputs_analysis_test3` 目录下将生成以下内容：

### 6.1 文件夹结构

- **`1_specularity_removed/`**:
  
  - 存放去除了反光点、修复后的干净舌象图片（.jpg）。

- **`2_subregions/`**:
  
  - 存放五分区（尖、根、中、左、右）的可视化 Mask 图（.png）。

- **`3_coat_body/`**:
  
  - `*_body.png`: 舌质区域二值 Mask。
  
  - `*_coat.png`: 舌苔区域二值 Mask。
  
  - `*_vis.jpg`: 舌质（红）与舌苔（绿）的叠加可视化图。

- **`4_reports/`**:
  
  - `*_report.jpg`: 2x2 的综合拼图，包含原图轮廓、去高光图、分区图和苔质分离图，方便人工质检。

### 6.2 数据汇总表 (`processing_summary.csv`)

该表格包含每张图片的关键定量指标：

| **字段名**             | **含义**    | **备注**               |
| ------------------- | --------- | -------------------- |
| **Filename**        | 样本 ID     | 如 `0001`             |
| **Tongue_Area_Px**  | 全舌像素面积    |                      |
| **Coat_Area_Px**    | 舌苔像素面积    |                      |
| **Body_Area_Px**    | 舌质像素面积    |                      |
| **Coat_Ratio**      | 舌苔占比      | $Coat / Total$ (0~1) |
| **Body_Ratio**      | 舌质占比      | $Body / Total$ (0~1) |
| **Gray_Tip**        | 舌尖区域灰度中位数 | 反映舌尖颜色深浅             |
| **Gray_Root**       | 舌根区域灰度中位数 | 反映舌根苔厚度              |
| **Gray_Left/Right** | 左右边缘灰度中位数 | 用于验证对称性              |
| **Gray_Center**     | 舌中区域灰度中位数 |                      |

---

## 7. 算法逻辑说明

以下是各步骤的核心算法逻辑：

1. **Mask 对齐 (Alignment)**:
   
   - 先计算 Mask 的**几何重心 (Centroid)**。
   
   - 以此重心为锚点，应用 JSON 中的缩放和旋转参数，最后叠加平移量 `tx, ty`。

2. **去高光 (De-specularity)**:
   
   - 使用 **CLAHE** (限制对比度自适应直方图均衡化) 增强图像。
   
   - 阈值 (`>230`) 提取高光，膨胀 Mask 后使用 **Telea 算法** (`cv2.inpaint`) 进行修复。

3. **分区 (Sub-regions)**:
   
   - 严格复现论文标准：
     
     - **Tip (舌尖)**: 底部 20%
     
     - **Root (舌根)**: 顶部 20%
     
     - **Margins (边缘)**: 左右各 20%
     
     - **Center (舌中)**: 剩余中心区域

4. **苔质分离 (Clustering)**:
   
   - 转换色彩空间至 **Lab** (取 a, b 通道) 和 **HSV** (取 S 通道)。
   
   - 特征向量：`[2.0*a, 0.8*b, 1.2*S]`。
   
   - 使用 **K-Means (k=2)** 聚类，根据“红度+饱和度”评分自动判定哪个簇是舌质，哪个是舌苔。

## 8. 统计学验证与高级分析

完成批处理并生成 `processing_summary.csv` 后，使用 `statistical_analysis.py` 进行自动化假设检验，以验证数据的生理学合理性。

### 8.1 功能概述

该模块复现了 *TonguExpert* 论文中的统计验证逻辑：

1. **正态性检验**：自动判断数据分布，智能选择 **配对 T 检验** 或 **Wilcoxon 符号秩检验**。

2. **生理规律验证**：
   
   - **对称性验证**：检验左边缘 (Left) 与右边缘 (Right) 灰度是否一致（预期：无显著差异 $P>0.05$）。
   
   - **特异性验证**：检验舌尖 (Tip) 与舌根 (Root) 灰度是否差异显著（预期：极显著差异 $P<0.001$）。

3. **多维可视化**：生成带有显著性标注的箱线图和相关性热力图。

### 8.2 运行步骤

确保 `outputs_analysis_test3/processing_summary.csv` 已经存在，然后在终端运行：

```Bash
python statistical_analysis.py
```

### 8.3 分析结果解读 (Outputs)

结果将保存在 `outputs_analysis_test3/5_hypothesis_testing/` 目录下：

#### A. 统计报表 (`hypothesis_test_results.csv`)

这是核心结论表，包含以下关键列：

- **Comparison**: 对比组（如 `Left Margin vs Right Margin`）。

- **Test_Method**: 使用的检验方法（如 `Paired T-Test`）。

- **P_Value**: 统计学显著性概率。

- **Conclusion**: 结论判读（`Accept H0` 代表无差异，`Reject H0` 代表有显著差异）。

#### B. 描述性统计 (`descriptive_statistics.csv`)

包含所有指标（灰度、占比、色彩均值）的 **均值 (mean)**、**标准差 (std)** 和 **四分位数**，可直接用于论文中的 "Table 1: Patient Characteristics"。

#### C. 论文级图表

1. **`boxplot_significance.png` (显著性箱线图)**
   
   - **看点**：图上方带有“门框线”和星号（`***` 或 `ns`）。
   
   - **验证标准**：
     
     - `Left` 和 `Right` 之间的连线应标注为 `ns` (No Significance)。
     
     - `Tip` 和 `Root` 之间的连线应标注为 `***` (Highly Significant)。

2. **`heatmap_correlation.png` (相关性热力图)**
   
   - **看点**：展示 **舌苔占比 (Coat_Ratio)** 与 **亮度 (L)** 呈正相关（红色），与 **红度 (a)** 呈负相关（蓝色），符合“苔多色白”的中医理论。
