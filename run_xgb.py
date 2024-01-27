import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

start_time = time.time()

# 1. 巧妙利用你已经规范化好的 CSV 数据集（比直接读 .dat 更稳）
print("正在加载数据集...")
train_dataSet = pd.read_csv("./dataset/my_dataset/train_data.csv")
# 确保使用的是包含真实值的 662 本地验证集
test_dataSet = pd.read_csv("./dataset/my_dataset/test_data.csv")

# 2. 声明特征列与目标列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 3. 划分 X 和 y
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]

# 4. 初始化原代码指定的参数
other_params = {
    'seed': 217,
    'booster': 'gbtree',
    'max_depth': 2,
    'n_estimators': 120,
    'learning_rate': 0.1,
    'gamma': 5,
    'reg_alpha': 50,
    'reg_lambda': 30,
    'min_child_weight': 20,
    'colsample_bytree': 0.4,
    'subsample': 0.5,
}
model_adj = XGBRegressor(**other_params)

# 5. 模型训练与预测
print("XGBoost 开始训练...")
model_adj.fit(X_train, y_train)

print("XGBoost 开始推理...")
y_predict = model_adj.predict(X_test)

# 6. 导出大作业专用的结果格式
results = []
for Predicted_Value in y_predict:
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    results.append([formatted_predicted_value])

result_df = pd.DataFrame(results, columns=['Predicted_Value'])
result_df.to_csv("result_XGB.csv", index=False)

print("🎉 预测结果已成功保存至 result_XGB.csv")
print(f"总耗时：{time.time() - start_time : .3f}秒")