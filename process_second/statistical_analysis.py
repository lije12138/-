# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# 忽略一些绘图警告
warnings.filterwarnings("ignore")

# --- 绘图配置 (支持中文) ---
def config_fonts():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'Microsoft YaHei', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    sns.set(style="whitegrid", font='SimHei') 

class HypothesisTester:
    def __init__(self, 
                 csv_path="outputs_analysis_test3/processing_summary.csv",
                 output_dir="outputs_analysis_test3/5_hypothesis_testing"):
        
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config_fonts()

    def load_data(self):
        if not self.csv_path.exists():
            print(f"❌ 错误：找不到文件 {self.csv_path}")
            return None
        
        df = pd.read_csv(self.csv_path)
        print(f"✅ 数据加载成功: {len(df)} 样本")
        
        # 简单清洗: 确保灰度值 > 0
        df = df[df['Gray_Tip'] > 0].copy()
        return df

    def check_normality(self, data):
        """进行 Shapiro-Wilk 正态性检验"""
        # 如果样本量 > 5000, Shapiro 可能会过于敏感，通常看 Q-Q 图，这里为了自动化使用 Shapiro
        # 为了性能，如果数据量太大，可以抽样，但这里跑全量
        stat, p = stats.shapiro(data)
        return p > 0.05 # True 表示服从正态分布

    def perform_paired_test(self, group1, group2, name1, name2):
        """自动选择 T检验 或 Wilcoxon检验"""
        # 1. 检查正态性 (差值是否正态)
        diff = group1 - group2
        is_normal = self.check_normality(diff)
        
        test_method = ""
        stat_val = 0
        p_val = 0
        
        if is_normal:
            test_method = "Paired T-Test (配对T检验)"
            stat_val, p_val = stats.ttest_rel(group1, group2)
        else:
            test_method = "Wilcoxon Signed-Rank (威尔科克森符号秩)"
            stat_val, p_val = stats.ranksums(group1, group2) # 或者 wilcoxon
            # 注意: ranksums 是秩和，wilcoxon 是符号秩。配对数据通常用 wilcoxon
            try:
                stat_val, p_val = stats.wilcoxon(group1, group2)
            except:
                # 处理全0差值的情况
                p_val = 1.0
        
        # 判定显著性
        significance = ""
        if p_val < 0.001: significance = "*** (极显著)"
        elif p_val < 0.01: significance = "** (显著)"
        elif p_val < 0.05: significance = "* (差异)"
        else: significance = "ns (无差异)"
        
        return {
            "Comparison": f"{name1} vs {name2}",
            "Test_Method": test_method,
            "Statistic": round(stat_val, 4),
            "P_Value": p_val, # 保留原始精度用于科学计数法
            "P_Label": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
            "Significance": significance,
            "Normality_Assumption": "Satisfied" if is_normal else "Violated"
        }

    def run_full_analysis(self):
        df = self.load_data()
        if df is None: return

        results = []
        
        # ==========================================
        # 1. 区域灰度特征检验 (复现论文核心假设)
        # ==========================================
        print("\n 正在进行区域灰度假设检验...")
        
        # 假设1: 对称性 (Left vs Right) -> 期望无差异
        results.append(self.perform_paired_test(df['Gray_Left'], df['Gray_Right'], "Left Margin", "Right Margin"))
        
        # 假设2: 生理梯度 (Tip vs Root) -> 期望有差异
        results.append(self.perform_paired_test(df['Gray_Tip'], df['Gray_Root'], "Tongue Tip", "Tongue Root"))
        
        # 假设3: 生理梯度 (Tip vs Center) -> 期望有差异
        results.append(self.perform_paired_test(df['Gray_Tip'], df['Gray_Center'], "Tongue Tip", "Tongue Center"))
        
        # 假设4: 生理梯度 (Root vs Center) -> 期望有差异
        results.append(self.perform_paired_test(df['Gray_Root'], df['Gray_Center'], "Tongue Root", "Tongue Center"))

        # ==========================================
        # 2. 导出统计结果 CSV
        # ==========================================
        res_df = pd.DataFrame(results)
        # 添加人类可读结论列
        res_df['Conclusion'] = res_df['P_Value'].apply(lambda p: "Reject H0 (有差异)" if p < 0.05 else "Accept H0 (无差异)")
        
        csv_save_path = self.output_dir / "hypothesis_test_results.csv"
        res_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig') # sig 解决Excel中文乱码
        print(f" 检验结果已保存: {csv_save_path}")
        
        # ==========================================
        # 3. 描述性统计摘要
        # ==========================================
        desc_cols = ['Gray_Tip', 'Gray_Root', 'Gray_Left', 'Gray_Right', 'Gray_Center', 
                     'Coat_Ratio', 'Color_Mean_L', 'Color_Mean_a']
        # 确保列存在
        valid_desc_cols = [c for c in desc_cols if c in df.columns]
        desc_df = df[valid_desc_cols].describe().T
        desc_df['iqr'] = desc_df['75%'] - desc_df['25%'] # 四分位距
        desc_save_path = self.output_dir / "descriptive_statistics.csv"
        desc_df.to_csv(desc_save_path, encoding='utf-8-sig')
        print(f" 描述性统计已保存: {desc_save_path}")

        # ==========================================
        # 4. 绘图 (带显著性标注)
        # ==========================================
        self.plot_significance_boxplot(df, results)
        self.plot_correlation(df)
        
    def plot_significance_boxplot(self, df, test_results):
        """绘制带有P值标注的箱线图"""
        print(" 正在绘制箱线图...")
        cols = ['Gray_Tip', 'Gray_Root', 'Gray_Center', 'Gray_Left', 'Gray_Right']
        if not all(c in df.columns for c in cols): return

        # 转换数据格式
        plot_data = df.melt(value_vars=cols, var_name='Region', value_name='Grayscale')
        plot_data['Region'] = plot_data['Region'].str.replace('Gray_', '')
        
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x='Region', y='Grayscale', data=plot_data, palette="Set3", width=0.6)
        sns.stripplot(x='Region', y='Grayscale', data=plot_data, color=".3", alpha=0.3, size=2)
        
        # 获取最大Y值用于画线
        y_max = plot_data['Grayscale'].max()
        h = y_max * 0.05 
        
        # 手动标注显著性 (根据 test_results)
        # 这里为了美观，我们只标注 Tip vs Root 和 Left vs Right
        
        # 1. 标注 Tip vs Root
        # 找到 Tip 和 Root 的 x 坐标 (Tip=0, Root=1, Center=2, Left=3, Right=4 - 取决于melt顺序)
        # melt顺序通常是列表顺序: Tip, Root, Center, Left, Right
        x1, x2 = 0, 1 # Tip, Root
        p_val = next((r['P_Value'] for r in test_results if "Tip" in r['Comparison'] and "Root" in r['Comparison']), 1)
        sig_symbol = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        
        self.add_stat_annotation(ax, x1, x2, y_max + h, h, sig_symbol)

        # 2. 标注 Left vs Right
        x1, x2 = 3, 4 # Left, Right
        p_val = next((r['P_Value'] for r in test_results if "Left" in r['Comparison'] and "Right" in r['Comparison']), 1)
        sig_symbol = "ns" if p_val >= 0.05 else "**"
        
        self.add_stat_annotation(ax, x1, x2, y_max + h, h, sig_symbol)

        plt.title('舌象五分区灰度分布及显著性检验', fontsize=16)
        plt.ylabel('灰度中位数 (0-255)', fontsize=12)
        plt.xlabel('解剖学分区', fontsize=12)
        
        save_path = self.output_dir / "boxplot_significance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")

    def add_stat_annotation(self, ax, x1, x2, y, h, text):
        """辅助函数：画横线和星星"""
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color='k', fontsize=12)

    def plot_correlation(self, df):
        """绘制增强版相关性热力图"""
        print(" 正在绘制相关性热力图...")
        target_cols = ['Coat_Ratio', 'Body_Ratio', 'Shape_AspectRatio', 
                       'Color_Mean_L', 'Color_Mean_a', 'Color_Mean_b',
                       'Gray_Tip', 'Gray_Root']
        
        valid_cols = [c for c in target_cols if c in df.columns]
        if len(valid_cols) < 2: return
        
        corr = df[valid_cols].corr()
        
        # 汉化标签以便展示
        col_map = {
            'Coat_Ratio': '舌苔占比',
            'Body_Ratio': '舌质占比',
            'Shape_AspectRatio': '长宽比',
            'Color_Mean_L': '亮度(L)',
            'Color_Mean_a': '红度(a)',
            'Color_Mean_b': '黄度(b)',
            'Gray_Tip': '舌尖灰度',
            'Gray_Root': '舌根灰度'
        }
        corr.rename(index=col_map, columns=col_map, inplace=True)
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool)) # 只显示下半三角
        
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', 
                    vmax=1, vmin=-1, center=0, square=True, linewidths=.5,
                    cbar_kws={"shrink": .7})
        
        plt.title('多维特征相关性分析 (Pearson Correlation)', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        
        save_path = self.output_dir / "heatmap_correlation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")

if __name__ == "__main__":
    # 请确保路径与之前一致
    tester = HypothesisTester(
        csv_path="outputs_analysis_test3/processing_summary.csv",
        output_dir="outputs_analysis_test3/5_hypothesis_testing"
    )
    tester.run_full_analysis()