import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 加载两个模型在无真实值 662 盲测集上吐出的 CSV
xgb_path = './result_XGB.csv'
dlinear_path = './no_real_val_result_LTSF_Linear.csv'

print("正在载入盲测集预测结果...")
xgb_df = pd.read_csv(xgb_path)
dl_df = pd.read_csv(dlinear_path)

# 2. 解析空格分隔的字符串矩阵
xgb_vals = np.array(xgb_df['Predicted_Value'].apply(lambda x: list(map(float, x.split()))).tolist())
dl_vals = np.array(dl_df['Predicted_Value'].apply(lambda x: list(map(float, x.split()))).tolist())

target_columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']

# 3. 核心核查一：统计分布核查 (Statistical Sanity Check)
print("\n" + "="*65)
print(f"{'特征目标':<22} | {'模型':<8} | {'最小值':<8} | {'最大值':<8} | {'平均值':<8}")
print("-"*65)
for i, col in enumerate(target_columns):
    print(f"{col:<22} | {'XGB':<8} | {xgb_vals[:, i].min():<8.2f} | {xgb_vals[:, i].max():<8.2f} | {xgb_vals[:, i].mean():<8.2f}")
    print(f"{col:<22} | {'DLinear':<8} | {dl_vals[:, i].min():<8.2f} | {dl_vals[:, i].max():<8.2f} | {dl_vals[:, i].mean():<8.2f}")
    print("-"*65)

# 4. 核心核查二：计算双模型预测的相关系数 (Consistency Check)
print("\n" + "="*45)
print(f"{'特征目标':<22} | {'双模型预测相关性 (Correlation)'}")
print("-"*45)
for i, col in enumerate(target_columns):
    corr = np.corrcoef(xgb_vals[:, i], dl_vals[:, i])[0, 1]
    print(f"{col:<22} | {corr:.4f}")
print("="*45)

# 5. 核心核查三：局部时序趋势可视化 (Visual Check)
print("\n正在绘制前 500 个时间步的时序去噪拟合比对图...")
plt.figure(figsize=(14, 10))
for i, col in enumerate(target_columns):
    plt.subplot(3, 2, i+1)
    # 截取前 500 个点观察局部微观物理量变化
    plt.plot(xgb_vals[:500, i], label='XGBoost', color='#E64B35', alpha=0.6, linestyle='--')
    plt.plot(dl_vals[:500, i], label='DLinear (Ours)', color='#4DBBD5', alpha=0.8)
    plt.title(f'{col} Blind Test Preview (First 500 Steps)', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    if i == 0:
        plt.legend(fontsize=8)

plt.tight_layout()
output_img = './blind_test_sanity_check.png'
plt.savefig(output_img, dpi=300)
print(f"🎉 核查完毕！微观时序对比图已导出至: {output_img}\n")