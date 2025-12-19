import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
# Mask文件目录
MASK_ROOT_DIR = "/hdd10Tb/zhangyi/SegChange-R1/outputs/change_41"
# TP/FP目录（真实标签）
TP_DIR = "/hdd10Tb/zhangyi/ChangeDetection/select_by_rf_09_v3_png_select_pngs/TP"
FP_DIR = "/hdd10Tb/zhangyi/ChangeDetection/select_by_rf_09_v3_png_select_pngs/FP"

# 标签映射
LABEL_CHANGE = 1       # 变化
LABEL_NO_CHANGE = 0    # 无变化

# 可视化配置
PLOT_SAVE_PATH = "/hdd10Tb/zhangyi/SegChange-R1/outputs/change_41/confusion_matrix_heatmap.png"
PIXEL_PLOT_DIR = "/hdd10Tb/zhangyi/SegChange-R1/outputs/change_41/pixel_plots"
PLOT_FONT_SIZE = 12
PLOT_DPI = 300
PLOT_CMAP = "Blues"

# 创建目录
os.makedirs(PIXEL_PLOT_DIR, exist_ok=True)

# ===================== 核心工具函数 =====================
def extract_id_from_mask_filename(filename):
    """从mask文件名提取核心ID"""
    match = re.search(r'id([0-9a-f]+)_\d+_result_mask', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    match_backup = re.search(r'id([0-9a-f]+)_', filename, re.IGNORECASE)
    if match_backup:
        return match_backup.group(1)
    print(f"⚠️ 无法提取ID：{filename}")
    return None

def get_true_label_by_id(img_id):
    """根据ID获取真实标签（TP=变化，FP=无变化）"""
    tp_path = os.path.join(TP_DIR, img_id)
    fp_path = os.path.join(FP_DIR, img_id)
    if os.path.exists(tp_path) and os.path.isdir(tp_path):
        return LABEL_CHANGE
    elif os.path.exists(fp_path) and os.path.isdir(fp_path):
        return LABEL_NO_CHANGE
    else:
        print(f"⚠️ ID {img_id} 未在TP/FP目录找到")
        return None

def get_pred_info_from_mask(mask_file_path):
    """
    读取mask文件，返回：
    - 预测标签（1=变化，0=无变化）
    - 变化像素数（统计>0的像素，而非仅==1）
    """
    try:
        with Image.open(mask_file_path) as mask_img:
            mask_array = np.array(mask_img, dtype=np.uint8)
            # 核心修改：统计所有>0的像素数
            change_pixel_count = np.sum(mask_array > 0)  
            pred_label = LABEL_CHANGE if change_pixel_count > 0 else LABEL_NO_CHANGE
            return pred_label, change_pixel_count
    except Exception as e:
        print(f"⚠️ 读取mask失败 {mask_file_path}：{str(e)[:50]}")
        return None, None

def collect_all_mask_files(mask_root_dir):
    """收集所有*_result_mask.tif文件"""
    mask_files = []
    if not os.path.exists(mask_root_dir):
        print(f"❌ Mask目录不存在：{mask_root_dir}")
        return mask_files
    for file_name in os.listdir(mask_root_dir):
        file_path = os.path.join(mask_root_dir, file_name)
        if (os.path.isfile(file_path) and 
            file_name.lower().endswith('.tif') and 
            '_result_mask' in file_name):
            mask_files.append(file_name)
    print(f"✅ 收集到 {len(mask_files)} 个mask文件")
    return mask_files

# ===================== 可视化函数 =====================
def plot_confusion_matrix_heatmap(cm, target_names):
    """绘制混淆矩阵热力图"""
    plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=PLOT_DPI)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=PLOT_CMAP, ax=ax, cbar=True,
        xticklabels=target_names, yticklabels=target_names,
        linewidths=0.5, linecolor="gray", annot_kws={"size": PLOT_FONT_SIZE + 2}
    )
    ax.set_xlabel("Predicted Label", fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel("True Label", fontsize=PLOT_FONT_SIZE)
    ax.set_title("Confusion Matrix", fontsize=PLOT_FONT_SIZE + 4, pad=20)
    ax.tick_params(axis='x', labelsize=PLOT_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=PLOT_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 混淆矩阵热力图已保存：{os.path.abspath(PLOT_SAVE_PATH)}")

def print_pixel_statistics(tp_pixel_counts, fp_pixel_counts):
    """打印像素数统计信息"""
    print("\n" + "="*80)
    print("📊 像素数统计分析（仅统计>0的像素）")
    print("="*80)
    
    # TP统计
    if len(tp_pixel_counts) > 0:
        print(f"\n🔹 TP (真实变化+预测变化)：")
        print(f"   样本数：{len(tp_pixel_counts)}")
        print(f"   均值：{np.mean(tp_pixel_counts):.2f} | 中位数：{np.median(tp_pixel_counts):.2f}")
        print(f"   最小值：{np.min(tp_pixel_counts)} | 最大值：{np.max(tp_pixel_counts)}")
        print(f"   标准差：{np.std(tp_pixel_counts):.2f}")
        print(f"   四分位数：{np.percentile(tp_pixel_counts, [25, 50, 75])}")
    else:
        print(f"\n🔹 TP：无有效样本")
    
    # FP统计
    if len(fp_pixel_counts) > 0:
        print(f"\n🔹 FP (真实无变化+预测变化)：")
        print(f"   样本数：{len(fp_pixel_counts)}")
        print(f"   均值：{np.mean(fp_pixel_counts):.2f} | 中位数：{np.median(fp_pixel_counts):.2f}")
        print(f"   最小值：{np.min(fp_pixel_counts)} | 最大值：{np.max(fp_pixel_counts)}")
        print(f"   标准差：{np.std(fp_pixel_counts):.2f}")
        print(f"   四分位数：{np.percentile(fp_pixel_counts, [25, 50, 75])}")
    else:
        print(f"\n🔹 FP：无有效样本")
    print("="*80)

def plot_pixel_distribution(tp_pixel_counts, fp_pixel_counts):
    """绘制像素分布图表（箱型图+直方图）"""
    plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 数据准备
    data = [tp_pixel_counts, fp_pixel_counts]
    labels = ["TP (Change+Pred Change)", "FP (No Change+Pred Change)"]
    # 过滤空数据
    valid_data = [(d, l) for d, l in zip(data, labels) if len(d) > 0]
    if not valid_data:
        print("⚠️ 无有效像素数据用于绘图")
        return
    
    # 1. 箱型图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=PLOT_DPI)
    sns.boxplot(
        data=[d for d, _ in valid_data],
        ax=ax,
        palette=["#2ecc71", "#e74c3c"][:len(valid_data)],
        width=0.5,
        linewidth=1.5
    )
    # 标注中位数
    for i, (d, _) in enumerate(valid_data):
        median = np.median(d)
        ax.text(i, median, f'Median: {int(median)}', 
                ha='center', va='center', fontsize=PLOT_FONT_SIZE-2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax.set_xticklabels([l for _, l in valid_data], fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel("Number of Pixels > 0", fontsize=PLOT_FONT_SIZE)
    ax.set_title("Pixel Count Distribution (Boxplot)", fontsize=PLOT_FONT_SIZE + 4, pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    box_path = os.path.join(PIXEL_PLOT_DIR, "pixel_boxplot.png")
    plt.savefig(box_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 箱型图已保存：{os.path.abspath(box_path)}")
    
    # 2. 直方图+密度曲线
    fig, ax = plt.subplots(figsize=(12, 6), dpi=PLOT_DPI)
    colors = ["#2ecc71", "#e74c3c"]
    for i, (d, l) in enumerate(valid_data):
        sns.histplot(
            d, ax=ax, color=colors[i], label=l,
            kde=True, bins=30, alpha=0.7
        )
    ax.set_xlabel("Number of Pixels > 0", fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel("Sample Count", fontsize=PLOT_FONT_SIZE)
    ax.set_title("Pixel Count Distribution (Histogram+KDE)", fontsize=PLOT_FONT_SIZE + 4, pad=20)
    ax.legend(fontsize=PLOT_FONT_SIZE)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    hist_path = os.path.join(PIXEL_PLOT_DIR, "pixel_histogram.png")
    plt.savefig(hist_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 直方图已保存：{os.path.abspath(hist_path)}")

# ===================== 主逻辑 =====================
def main():
    """核心流程：数据收集 → 混淆矩阵 → 像素统计 → 可视化"""
    # 初始化变量
    true_labels = []
    pred_labels = []
    tp_pixel_counts = []  # TP样本的>0像素数
    fp_pixel_counts = []  # FP样本的>0像素数
    valid_count = 0
    invalid_count = 0
    
    # 1. 收集mask文件
    mask_files = collect_all_mask_files(MASK_ROOT_DIR)
    if not mask_files:
        print("❌ 无符合条件的mask文件")
        return
    
    # 2. 处理每个mask文件
    print("\n📝 开始处理mask文件并统计像素...")
    for file_name in mask_files:
        # 提取ID
        img_id = extract_id_from_mask_filename(file_name)
        if not img_id:
            invalid_count += 1
            continue
        
        # 获取真实标签
        true_label = get_true_label_by_id(img_id)
        if true_label is None:
            invalid_count += 1
            continue
        
        # 获取预测标签和像素数（>0）
        mask_path = os.path.join(MASK_ROOT_DIR, file_name)
        pred_label, pixel_count = get_pred_info_from_mask(mask_path)
        if pred_label is None or pixel_count is None:
            invalid_count += 1
            continue
        
        # 记录基础标签
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        valid_count += 1
        
        # 仅统计预测变化（像素>0）的样本
        if pred_label == LABEL_CHANGE and pixel_count > 0:
            if true_label == LABEL_CHANGE:
                tp_pixel_counts.append(pixel_count)  # TP样本
            else:
                fp_pixel_counts.append(pixel_count)  # FP样本
    
    # 3. 校验有效样本
    if valid_count == 0:
        print("❌ 无有效样本")
        return
    
    # 4. 计算混淆矩阵并绘图
    target_names = ["Change", "No Change"]
    cm = confusion_matrix(true_labels, pred_labels, labels=[LABEL_CHANGE, LABEL_NO_CHANGE])
    plot_confusion_matrix_heatmap(cm, target_names)
    
    # 5. 像素统计与可视化
    print_pixel_statistics(tp_pixel_counts, fp_pixel_counts)
    plot_pixel_distribution(tp_pixel_counts, fp_pixel_counts)
    
    # 6. 计算分类指标
    cls_report = classification_report(
        true_labels, pred_labels,
        labels=[LABEL_CHANGE, LABEL_NO_CHANGE],
        target_names=target_names,
        digits=4
    )
    
    # 7. 输出混淆矩阵和分类指标
    print("\n" + "="*60)
    print("📊 混淆矩阵与分类指标")
    print("="*60)
    print("\n🔍 混淆矩阵（行=真实标签，列=预测标签）：")
    print(f"                预测：变化      预测：无变化")
    print(f"真实：变化       {cm[0][0]}            {cm[0][1]}")
    print(f"真实：无变化     {cm[1][0]}            {cm[1][1]}")
    
    # 关键指标计算
    tp = cm[0][0]
    tn = cm[1][1]
    fp = cm[1][0]
    fn = cm[0][1]
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n📈 关键指标：")
    print(f"准确率(Accuracy)：{accuracy:.4f}")
    print(f"精确率(Precision)：{precision:.4f}")
    print(f"召回率(Recall)：{recall:.4f}")
    print(f"F1分数(F1-Score)：{f1:.4f}")
    
    print("\n📋 详细分类报告：")
    print(cls_report)
    
    print("\n📌 样本统计：")
    print(f"有效样本数：{valid_count}")
    print(f"无效样本数：{invalid_count}")
    print(f"TP像素样本数：{len(tp_pixel_counts)}")
    print(f"FP像素样本数：{len(fp_pixel_counts)}")
    print(f"总mask文件数：{len(mask_files)}")
    print("="*60)

# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 依赖检查
    try:
        import PIL
        import numpy
        import seaborn
        import matplotlib
        from sklearn import metrics
    except ImportError as e:
        print(f"❌ 缺少依赖库：pip install pillow numpy scikit-learn seaborn matplotlib")
        print(f"   缺失：{e.name}")
        exit(1)
    
    # 打印配置
    print("🔧 运行配置：")
    print(f"   Mask目录：{MASK_ROOT_DIR}")
    print(f"   TP目录：{TP_DIR} | FP目录：{FP_DIR}")
    print(f"   像素统计规则：统计mask中所有>0的像素数（而非仅==1）")
    
    # 执行主流程
    main()