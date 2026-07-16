# import numpy as np

# # 1. 读取现有的 npz 文件
# npz_path = "./embeddings/gene_embeddingss_full_common.npz"
# data = np.load(npz_path, allow_pickle=True)

# print("===== 正在精准对齐基因嵌入矩阵 =====")

# # 2. 明确提取 'gene_names' 和 'embeddings'
# gene_names = data['gene_names']
# embeddings = data['embeddings']

# print(f"总计包含基因数量: {len(gene_names)}")
# print(f"原生特征矩阵的形状 (Shape): {embeddings.shape}")

# # 3. 转存为模型死等的文件名
# output_path = "./embeddings/gene_embeddings_norman_512.npy"
# np.save(output_path, embeddings)

# print(f"【成功】已将 {embeddings.shape} 的真实特征矩阵桥接至: {output_path}")



import numpy as np
import pickle
import os

print("===== 开始进行单细胞微扰基因名绝对对齐 =====")

# 1. 载入你手头的原生特征和基因名
npz_path = "./embeddings/gene_embeddingss_full_common.npz"
raw_data = np.load(npz_path, allow_pickle=True)
my_gene_names = [str(g).upper() for g in raw_data['gene_names']]  # 统一转大写防错
my_embeddings = raw_data['embeddings']

# 建立 基因名 -> 原矩阵行索引 的极速查找字典
gene_to_idx = {name: idx for idx, name in enumerate(my_gene_names)}

# 2. 从封装好的 cell_graphs.pkl 中强行打捞模型死等的基因列表
pyg_data_path = "./data/norman/data_pyg/cell_graphs.pkl"
print(f"正在读取 PyG 封装库: {pyg_data_path} ...")

with open(pyg_data_path, 'rb') as f:
    cell_graphs_data = pickle.load(f)

# 很多基于 PyG 框架的生信模型，会将基因列表挂在字典的 'gene_list' 或 'pyg_obj' 的 var 空间里
# 我们来探测它存放的 Key
if isinstance(cell_graphs_data, dict):
    if 'gene_list' in cell_graphs_data:
        model_expected_genes = cell_graphs_data['gene_list']
    elif 'gene_names' in cell_graphs_data:
        model_expected_genes = cell_graphs_data['gene_names']
    else:
        # 兜底：如果它是一个 cell_id 为 key 的大字典，我们取出第一个 cell 的图对象来看看
        first_cell_key = list(cell_graphs_data.keys())[0]
        first_graph = cell_graphs_data[first_cell_key]
        if hasattr(first_graph, 'gene_list'):
            model_expected_genes = first_graph.gene_list
        elif 'gene_list' in first_graph:
            model_expected_genes = first_graph['gene_list']
        else:
            # 最后的极端情况：直接尝试去读 scPert 模型配置中硬编码的 5045
            print("未能直接解出基因列表键，尝试执行默认打捞...")
            model_expected_genes = cell_graphs_data.get('genes', [])
else:
    model_expected_genes = []

# 如果打捞出来的列表依然为空，我们给它一个更直接的策略：从模型初始化阶段直接拦截
if len(model_expected_genes) == 0:
    print("⚠️ 无法直接从 pkl 字典外壳获取基因名，正在启动‘模型字典反射机制’获取...")
    # 通过 scanpy 缓存直接抽取
    import scanpy as sc
    adata = sc.read_h5ad("./data/norman/norman.h5ad")
    model_expected_genes = list(adata.var_names)

print(f"【成功打捞】模型期望的基因数量: {len(model_expected_genes)}")

# 3. 按照模型死等的顺序，去我们的特征库里抽脂重组
aligned_embeddings = []
missing_count = 0
embedding_dim = my_embeddings.shape[1]

for gene in model_expected_genes:
    gene_upper = str(gene).upper()
    if gene_upper in gene_to_idx:
        aligned_embeddings.append(my_embeddings[gene_to_idx[gene_upper]])
    else:
        # 匹配失败的基因，用随机正态分布或全零向量冷启动兜底
        aligned_embeddings.append(np.random.normal(0, 0.1, size=(embedding_dim,)))
        missing_count += 1

aligned_embeddings = np.array(aligned_embeddings)

print(f"对齐完成！成功匹配: {len(model_expected_genes) - missing_count} 个基因")
print(f"未匹配（已进行冷启动初始化）: {missing_count} 个基因")
print(f"最终生成的对齐嵌入矩阵形状 (Shape): {aligned_embeddings.shape}")

# 4. 转存覆盖
output_path = "./embeddings/gene_embeddings_norman_512.npy"
np.save(output_path, aligned_embeddings)
print(f"【绝对对齐成功】对齐特征已注入: {output_path}")