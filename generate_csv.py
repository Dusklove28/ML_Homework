import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from exp.exp_main import Exp_Main


# 1. 配置与训练时完全一致的全局基础参数
class Configs:
    seq_len = 336
    label_len = 48
    pred_len = 96
    enc_in = 15
    dec_in = 15
    c_out = 6
    root_path = './dataset/my_dataset/'


configs = Configs()


# 2. 补全完整的参数Dummy类，彻底解决 use_gpu 缺失等所有 AttributeError
class ArgsDummy:
    def __init__(self):
        self.model = 'DLinear'
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.individual = True
        self.moving_avg = 25
        self.output_attention = False

        # 满血开启服务器 GPU 算力
        self.use_gpu = True if torch.cuda.is_available() else False
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'


# 3. 严格复刻 661 训练集的标准化基准（杜绝数据泄露，保障精度一致）
print("正在从 661 训练集构建标准化基准...")
df_train = pd.read_csv(os.path.join(configs.root_path, 'train_data.csv'))
num_train = int(len(df_train) * 0.8)

feature_cols = ['Ux', 'Uy', 'Uz', 'diag_sonic', 'diag_irga', 'T_SONIC_corr', 'TA_1_1_1', 'PA', 'FW',
                'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
target_cols = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']

train_x_raw = df_train[feature_cols].values[:num_train]
train_y_raw = df_train[target_cols].values[:num_train]

scaler_x = StandardScaler()
scaler_x.fit(train_x_raw)

scaler_y = StandardScaler()
scaler_y.fit(train_y_raw)


# 4. 读取 662 测试集并执行精准的时序行数对齐填充 (Padding)
print("正在读取并对齐测试集行数...")
df_test = pd.read_csv(os.path.join(configs.root_path, 'test_data_no_real_val.csv'))
test_x_raw = df_test[feature_cols].values
N = len(test_x_raw)

# 先用训练集基准进行特征标准化
test_x_scaled = scaler_x.transform(test_x_raw)

# 核心对齐逻辑：在测试集头部填充 seq_len 长度的边界数据
padding_rows = np.repeat(test_x_scaled[0:1], configs.seq_len, axis=0)
test_x_padded = np.vstack([padding_rows, test_x_scaled])

# 切片组装高维时序滑窗矩阵
windows = []
for i in range(N):
    windows.append(test_x_padded[i: i + configs.seq_len])


# 5. 加载全场最优 test_1 的权重路径
checkpoint_path = "./checkpoints/ML_final_DLinear_tuned_DLinear_custom_ftM_sl336_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_1/checkpoint.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"未找到权重文件：{checkpoint_path}，请核对 checkpoints 目录下的文件夹名称是否一致！")

args = ArgsDummy()
exp = Exp_Main(args)
print(f"模型加载成功，正在注入 DLinear 权重: {checkpoint_path}")
exp.model.load_state_dict(torch.load(checkpoint_path))
exp.model.eval()

device = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
exp.model.to(device)


# 6. 分批次进行前向传播回归预测
print("正在利用 GPU 进行高速去噪回归...")
X_tensor = torch.tensor(np.array(windows), dtype=torch.float32)

batch_size = 1024
all_preds = []

with torch.no_grad():
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_X = X_tensor[start_idx:end_idx].to(device)
        batch_out = exp.model(batch_X)  # [Batch, pred_len, c_out]

        # 提取未来预测的第一步（index 0）
        pred_single_step = batch_out[:, 0, :].cpu().numpy()
        all_preds.append(pred_single_step)


# 7. 逆标准化回真实的物理量绝对尺度
scaled_preds = np.vstack(all_preds)
true_scale_preds = scaler_y.inverse_transform(scaled_preds)

# 在 inverse_transform 之后，对真实物理尺度进行裁剪！
print("正在对预测结果进行物理边界合理性约束约束（消除非物理负数）...")
# 前 4 列（T_SONIC, CO2_density, CO2_density_fast_tmpr, H2O_density）浓度绝对不能为负
true_scale_preds[:, 0:4] = np.clip(true_scale_preds[:, 0:4], 0, None)
# 后 2 列（H2O_sig_strgth, CO2_sig_strgth）作为信号强度，范围死死限制在 0 到 1.0 之间
true_scale_preds[:, 4:6] = np.clip(true_scale_preds[:, 4:6], 0, 1.0)


# 8. 转换并导出符合大作业特殊要求的“空格分隔”提交文件
print("正在转换大作业规定的特殊字符串提交格式...")
string_preds = []
for row in true_scale_preds:
    # 将 6 个物理量预测值用空格拼接成一行长字符串
    str_val = " ".join([f"{x:.4f}" for x in row])
    string_preds.append(str_val)

# 构建最终提交的 DataFrame
submission_df = pd.DataFrame({
    'Predicted_Value': string_preds
})

output_csv = "./no_real_val_result_LTSF_Linear.csv"
submission_df.to_csv(output_csv, index=False)
print(f"经过物理边界优化的终极考卷已安全导出至: {output_csv}")