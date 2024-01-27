import pandas as pd
import os

# 1. 定义文件路径（请根据你的实际路径修改）
input_file = "modified_数据集Time_Series661.dat"
output_dir = "../dataset/my_dataset"
output_file = os.path.join(output_dir, "train_data.csv")

print("正在读取原始数据...")
# 原始数据是逗号分隔，直接用 read_csv 读取
df = pd.read_csv(input_file)

# 2. 结构规范化：将 TIMESTAMP 重命名为 date
if 'TIMESTAMP' in df.columns:
    df.rename(columns={'TIMESTAMP': 'date'}, inplace=True)
    print("成功将 'TIMESTAMP' 列重命名为 'date'。")
else:
    print("警告：未找到 'TIMESTAMP' 列，请检查数据源。")

# 3. 创建输出文件夹并保存
os.makedirs(output_dir, exist_ok=True)
df.to_csv(output_file, index=False)

print(f"规范化完成！新数据集已保存至: {output_file}")
print(f"当前数据集包含的特征列有: {list(df.columns[1:])}")
print(f"总特征维度（不含时间列）: {len(df.columns) - 1}")