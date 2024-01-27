import os
import pandas as pd
import numpy as np

# 显式配置 matplotlib 后端，防止在 Linux 服务器环境下引发缺少显示器的 GUI 报错
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 声明各个预测结果与带真实值的 662 本地验证集路径
xgb_file_path = './result_XGB.csv'
dlinear_file_path = './no_real_val_result_LTSF_Linear.csv'
true_file_path = "./dataset/my_dataset/test_data.csv"

# 2. 读取 CSV 数据集
print("正在读取模型预测结果与真实值标签...")
xgb_data = pd.read_csv(xgb_file_path)
dlinear_data = pd.read_csv(dlinear_file_path)
true_data = pd.read_csv(true_file_path)

# 提取 6 个期末大作业回归目标列
target_columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
true_values = true_data[target_columns].values

# 解析空格分隔的预测字符串高维矩阵
xgb_values = np.array(xgb_data['Predicted_Value'].apply(lambda x: list(map(float, x.split()))).tolist())
dlinear_values = np.array(dlinear_data['Predicted_Value'].apply(lambda x: list(map(float, x.split()))).tolist())

# 3. 严格执行行数匹配检查，防止数据泄露或滑窗对齐失误
assert len(xgb_values) == len(true_values), "❌ 严重错误：XGBoost 预测值和真实值的行数不匹配！"
assert len(dlinear_values) == len(true_values), "❌ 严重错误：DLinear 预测值和真实值的行数不匹配！"

# 4. 精准计算各特征的平均绝对误差（MAE）
xgb_errors = np.abs(xgb_values - true_values)
xgb_mean_errors = np.mean(xgb_errors, axis=0)
xgb_overall_mean_error = np.mean(xgb_errors)

dlinear_errors = np.abs(dlinear_values - true_values)
dlinear_mean_errors = np.mean(dlinear_errors, axis=0)
dlinear_overall_mean_error = np.mean(dlinear_errors)

# 5. 在 Linux 控制台中打印极为规整的学术对比报表
print("\n" + "="*70)
print(f"{'物理特征目标':<25} | {'XGBoost MAE':<12} | {'DLinear MAE':<12} | {'误差降低幅度 (%)':<10}")
print("-"*70)
for i, feature in enumerate(target_columns):
    # 计算当前特征深度学习对比传统机器学习的误差下降百分比
    improvement = (xgb_mean_errors[i] - dlinear_mean_errors[i]) / xgb_mean_errors[i] * 100
    print(f"{feature:<25} | {xgb_mean_errors[i]:<12.4f} | {dlinear_mean_errors[i]:<12.4f} | {improvement:<10.2f}%")
print("="*70)

overall_improvement = (xgb_overall_mean_error - dlinear_overall_mean_error) / xgb_overall_mean_error * 100
print(f"💡 XGBoost 基准模型总体平均绝对误差 (MAE): {xgb_overall_mean_error:.4f}")
print(f"💡 DLinear 时序解耦模型总体平均绝对误差 (MAE): {dlinear_overall_mean_error:.4f}")
print(f"🚀 深度学习算法相比于基准线整体精度提升幅度: {overall_improvement:.2f}%")
print("="*70 + "\n")

# 6. 生成高分辨率的双模型性能 PK 直方图 (柱状图)
print("正在绘制算法性能对比柱状图...")
x = np.arange(len(target_columns))
width = 0.35  # 柱子宽度

fig, ax = plt.subplots(figsize=(11, 6.5))
# 配色选用经典的学术对比色：红色代表经典机器学习基准，蓝色代表前沿深度学习
rects1 = ax.bar(x - width/2, xgb_mean_errors, width, label='XGBoost Baseline', color='#E64B35', edgecolor='black', alpha=0.85)
rects2 = ax.bar(x + width/2, dlinear_mean_errors, width, label='DLinear (Ours)', color='#4DBBD5', edgecolor='black', alpha=0.85)

# 美化图表元素
ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax.set_title(f'DLinear vs XGBoost Prediction Error Comparison\n(Overall Error Reduced by {overall_improvement:.2f}%)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(target_columns, rotation=15, fontsize=10, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 编写闭包自动化函数：为直方图顶部精准打上 3 位小数的数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 纵向偏移 4 个像素
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# 保存高质量图表
output_image_path = './algo_comparison_chart.png'
plt.savefig(output_image_path, dpi=300)
print(f"🎉 成功！高分辨率精度对比图表已导出至: {output_image_path}\n")