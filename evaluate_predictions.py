import pandas as pd
import numpy as np

pred_file_path = r'./result_XGB.csv'
# pred_file_path = r'./no_real_val_result_LTSF_Linear.csv'

true_file_path = r"./dataset/my_dataset/modified_数据集Time_Series662.dat"

pred_data = pd.read_csv(pred_file_path)
true_data = pd.read_csv(true_file_path)

target_columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
true_values = true_data[target_columns].values

pred_values = np.array(pred_data['Predicted_Value'].apply(lambda x: list(map(float, x.split()))).tolist())

assert len(pred_values) == len(true_values), "预测值和真实值的行数不匹配，请检查数据"

errors = np.abs(pred_values - true_values)
mean_errors = np.mean(errors, axis=0)
overall_mean_error = np.mean(errors)

print("每个特征的平均误差：")
for feature, error in zip(target_columns, mean_errors):
    print(f"{feature}: {error:.4f}")
print(f"\nXGB总体平均误差: {overall_mean_error:.4f}")
