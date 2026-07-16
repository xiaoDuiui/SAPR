# import os
# import sys
# import torch
# import numpy as np
# sys.path.append(r'./')
# from models import ProcePertdata,scpert


# torch.cuda.set_device('cuda:1')


# embedding_dir = "./embeddings/"

# # Get all embedding files
# embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]

# # Base data path
# data_path = "./data"
# embedding_to_data_map = {"gene_embeddings_norman_512.npy": "norman"}

# # Process each dataset
# for embedding_file, DataName in embedding_to_data_map.items():
#     print(f"\n===== Processing dataset: {DataName} with embedding: {embedding_file} =====\n")
    
#     # Initialize and prepare data
#     pertData = ProcePertdata.PertData(data_path)
#     pertData.load(DataName=DataName)
#     pertData.prepare_split(split='simulation', seed=77)
#     pertData.get_dataloader(batch_size=64, test_batch_size=64)
#     embedding_path = os.path.join(embedding_dir, embedding_file)
#     # Initialize scpert model
#     SCPert = scpert.scPert(pertData, device='cuda:1',
#                          weight_bias_track=False,
#                          proj_name='pertnet',
#                          exp_name=f'pertnet_{DataName}',
#                          embedding_path=embedding_path)
    
#     # Override the gene_emb attribute with the correct embedding file
    
#     SCPert.model_initialize(hidden_size=64)
    
#     # Load the correct embedding file
    
    
#     # Train the model
#     SCPert.train(epochs=25, lr=0.001)
#     # Save the model
#     SCPert.save_model(f'{DataName}_model_FINAL')
    
#     print(f"\n===== Completed dataset: {DataName} =====\n")

# print("All datasets processed successfully!")


import os
import sys
import torch
import numpy as np
sys.path.append(r'./')
from models import ProcePertdata, scpert

# 1. 自动适配显卡
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_device(device)
print(f"正在使用设备: {device}")

embedding_dir = "./embeddings/"
data_path = "./data"

# 原作者死等的文件名
embedding_file = "gene_embeddings_norman_512.npy"
DataName = "norman"

print(f"\n===== Processing dataset: {DataName} =====\n")

# 2. 初始化并加载单细胞微扰数据集
pertData = ProcePertdata.PertData(data_path)
pertData.load(DataName=DataName)
pertData.prepare_split(split='simulation', seed=77)
pertData.get_dataloader(batch_size=4, test_batch_size=4)

# ========================================================
# 核心大招：利用内存中的模型信息，现场打捞并绝对对齐你的特征！
# ========================================================
print("正在执行内存级基因特征绝对对齐（包含 Ensembl ID 翻译）...")
# 1. 捞出模型当前真正的基因列表（Ensembl ID 列表）
if hasattr(pertData, 'adata'):
    model_expected_genes = list(pertData.adata.var_names)
    # 核心映射：从单细胞数据 var 空间捞出 Ensembl ID -> Gene Symbol 的翻译字典
    # 统一转大写防止大小写不一致
    ensembl_to_symbol = {
        str(ens_id): str(sym).upper() 
        for ens_id, sym in zip(pertData.adata.var_names, pertData.adata.var['gene_name'])
    }
else:
    model_expected_genes = list(pertData.pert_names) if hasattr(pertData, 'pert_names') else []
    ensembl_to_symbol = {}

print(f"从内存成功拦截到模型期望的 Ensembl ID 数量: {len(model_expected_genes)}")

# 2. 载入你手头的原生 18101 维度的特征
npz_path = "./embeddings/gene_embeddingss_full_common.npz"
raw_data = np.load(npz_path, allow_pickle=True)
my_gene_names = [str(g).upper() for g in raw_data['gene_names']]
my_embeddings = raw_data['embeddings']
gene_to_idx = {name: idx for idx, name in enumerate(my_gene_names)}

aligned_embeddings = []
match_count = 0
embedding_dim = my_embeddings.shape[1]

# 3. 开始对齐：先翻译学号，再点名要特征
for ens_id in model_expected_genes:
    # 查字典：把 ENSGxxxx 翻译成类似 TSC22D1
    gene_symbol = ensembl_to_symbol.get(str(ens_id), str(ens_id).upper())
    
    if gene_symbol in gene_to_idx:
        # 匹配成功，抓取对应的向量
        aligned_embeddings.append(my_embeddings[gene_to_idx[gene_symbol]])
        match_count += 1
    else:
        # 未匹配到的用正态分布冷启动兜底
        aligned_embeddings.append(np.random.normal(0, 0.1, size=(embedding_dim,)))

aligned_embeddings = np.array(aligned_embeddings)
print(f"【对齐报告】成功通过 Gene Symbol 匹配: {match_count} / {len(model_expected_genes)} 个基因！")
print(f"未匹配（已执行冷启动兜底）: {len(model_expected_genes) - match_count} 个")

# 将对齐好的完美特征现场存入模型死等的路径
embedding_path = os.path.join(embedding_dir, embedding_file)
np.save(embedding_path, aligned_embeddings)
print(f"完美对齐特征已现场注入: {embedding_path}")
# ========================================================

# 3. 正式初始化微扰预测模型
SCPert = scpert.scPert(pertData, device=device,
                       weight_bias_track=False,
                       proj_name='pertnet',
                       exp_name=f'pertnet_{DataName}',
                       embedding_path=embedding_path)

SCPert.model_initialize(hidden_size=64)

print("【终极注入】正在将对齐特征强制覆盖进模型内存...")
SCPert.model.gene_emb = torch.nn.Parameter(torch.tensor(aligned_embeddings, dtype=torch.float32).to(device))

# 4. 开始训练
SCPert.train(epochs=25, lr=0.001)

# 5. 保存训练好的模型
SCPert.save_model(f'{DataName}_model_FINAL')
print(f"\n===== Completed dataset: {DataName} =====\n")
print("All datasets processed successfully!")