import os
import sys
import torch
import numpy as np
sys.path.append(r'./')
from models import ProcePertdata, scpert

# 【修改 1】将显卡改为默认的第一张显卡 cuda:0，适配单卡/WSL环境
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_device(device)

embedding_dir = "./embeddings/"

# Base data path
data_path = "./data"

# 【修改 2】根据你现有的 embedding 文件名进行映射 (假设使用 full_common 这个，或者你可以看具体文件名后缀)
# 注意：如果是 .npz 文件，请确保下游 scpert.py 里面支持 np.load 时读取格式；若是 .npy 正常填入即可。
# 我们这里先映射为你实际拥有的文件名：
embedding_to_data_map = {
    "gene_embeddingss_full_common.npz": "norman" 
}

# Process each dataset
for embedding_file, DataName in embedding_to_data_map.items():
    print(f"\n===== Processing dataset: {DataName} with embedding: {embedding_file} =====\n")

    # Initialize and prepare data
    pertData = ProcePertdata.PertData(data_path)
    pertData.load(DataName=DataName) # 这里触发自动下载或加载
    pertData.prepare_split(split='simulation', seed=77)
    pertData.get_dataloader(batch_size=64, test_batch_size=64)
    
    embedding_path = os.path.join(embedding_dir, embedding_file)
    
    # Initialize scpert model 【修改 3】传入上面定义好的动态 device
    SCPert = scpert.scPert(pertData, device=device,
                           weight_bias_track=False,
                           proj_name='pertnet',
                           exp_name=f'pertnet_{DataName}',
                           embedding_path=embedding_path)

    SCPert.model_initialize(hidden_size=64)

    # Train the model
    SCPert.train(epochs=25, lr=0.001)
    # Save the model
    SCPert.save_model(f'{DataName}_model_FINAL')

    print(f"\n===== Completed dataset: {DataName} =====\n")