import arff
import pandas as pd

# --- 加载 scene.arff 文件 ---
with open('scene.arff', 'r') as f:
    # 使用 return_type 强制返回结构化字典
    dataset = arff.load(f, return_type=arff.DENSE)

# --- 提取元数据和数据 ---
attributes = dataset['attributes']  # 所有列的定义（特征 + 标签）
data = dataset['data']              # 实际数据行

# --- 分离特征和标签 ---
# 前294列是特征，后6列是标签
features = [row[:294] for row in data]
labels = [row[294:] for row in data]

# --- 保存特征文件 ---
feature_names = [attr[0] for attr in attributes[:294]]  # 特征列名
df_features = pd.DataFrame(features)
df_features.to_csv("scene_data.tsv", sep='\t', index=False, header=False)

# --- 保存标签文件 ---
label_names = [attr[0] for attr in attributes[294:]]    # 标签列名
df_labels = pd.DataFrame(labels)
df_labels.to_csv("scene_target.tsv", sep='\t', index=False, header=False)

# 生成二进制编码（例如 101000 表示 beach + mountain）
df_labels['binary_code'] = df_labels.apply(
    lambda row: ''.join(row.astype(int).astype(str)),
    axis=1
)

# 转换为十进制整数（可选）
df_labels['numeric_code'] = df_labels['binary_code'].apply(
    lambda x: int(x, 2)
)

# 保存
df_labels[['numeric_code']].to_csv("scene_target_numeric.tsv", sep='\t', index=False, header=False)