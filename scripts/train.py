import os
import sys
import torch
import numpy as np
sys.path.append(r'./')
from models import ProcePertdata, scpert

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_device(device)
print(f'Using device: {device}')

embedding_dir = './embeddings/'
data_path = './data'
embedding_file = 'gene_embeddings_norman_512.npy'
DataName = 'norman'
npz_path = './embeddings/gene_embeddingss_full_common.npz'

print(f'\n===== Processing dataset: {DataName} =====\n')

pertData = ProcePertdata.PertData(data_path)
pertData.load(DataName=DataName)
pertData.prepare_split(split='simulation', seed=77)
pertData.get_dataloader(batch_size=4, test_batch_size=4)

print('Aligning embeddings via Ensembl ID translation...')
if hasattr(pertData, 'adata'):
    model_expected_genes = list(pertData.adata.var_names)
    ensembl_to_symbol = {
        str(ens_id): str(sym).upper()
        for ens_id, sym in zip(pertData.adata.var_names, pertData.adata.var['gene_name'])
    }
else:
    model_expected_genes = list(pertData.pert_names) if hasattr(pertData, 'pert_names') else []
    ensembl_to_symbol = {}

print(f'Expected Ensembl IDs: {len(model_expected_genes)}')

raw_data = np.load(npz_path, allow_pickle=True)
my_gene_names = [str(g).upper() for g in raw_data['gene_names']]
my_embeddings = raw_data['embeddings']
gene_to_idx = {name: idx for idx, name in enumerate(my_gene_names)}

aligned = []
match_count = 0
dim = my_embeddings.shape[1]

for ens_id in model_expected_genes:
    sym = ensembl_to_symbol.get(str(ens_id), str(ens_id).upper())
    if sym in gene_to_idx:
        aligned.append(my_embeddings[gene_to_idx[sym]])
        match_count += 1
    else:
        aligned.append(np.random.normal(0, 0.1, size=(dim,)))

aligned = np.array(aligned)
print(f'Aligned: {match_count} / {len(model_expected_genes)}, cold-start: {len(model_expected_genes) - match_count}')

embedding_path = os.path.join(embedding_dir, embedding_file)
np.save(embedding_path, aligned)
print(f'Embedding saved: {embedding_path}')

SCPert = scpert.scPert(pertData, device=device,
                       weight_bias_track=False,
                       proj_name='pertnet',
                       exp_name=f'pertnet_{DataName}',
                       embedding_path=embedding_path)

SCPert.model_initialize(
    hidden_size=64,
    use_deg_sparse=True,
    deg_sparse_ratio=0.25,
    deg_calibrated_lambda=0.0,
    interaction_lambda=0.05,
    deg_ratio=0.1
)

print('Injecting aligned embeddings...')
SCPert.model.gene_emb = torch.nn.Parameter(
    torch.tensor(aligned, dtype=torch.float32).to(device)
)

SCPert.train(epochs=25, lr=0.001)
SCPert.save_model(f'{DataName}_model_FINAL')
print(f'\n===== Completed: {DataName} =====\n')
print('All done!')
