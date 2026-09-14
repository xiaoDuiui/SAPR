
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', action='store_true', default=True,
                        help='Actually embed using sentence-transformers')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of GO neighbors per gene')
    parser.add_argument('--data_dir', type=str, default='./')
    args = parser.parse_args()

    print('[1/5] Loading gene list...')
    gene_data = np.load(f'{args.data_dir}embeddings/gene_2_kge_comgcn_final_common.npz', allow_pickle=True)
    gene_names = [str(g).upper() for g in gene_data['gene_names']]
    print(f'  Total genes: {len(gene_names)}')

    print('[2/5] Loading GO annotations...')
    go_df = pd.read_csv(f'{args.data_dir}data/norman/go.csv')
    go_neighbors = {}
    for _, row in go_df.iterrows():
        src = str(row['source']).upper()
        tgt = str(row['target']).upper()
        imp = row['importance']
        if src not in go_neighbors:
            go_neighbors[src] = []
        go_neighbors[src].append((tgt, imp))
    for g in go_neighbors:
        go_neighbors[g].sort(key=lambda x: -x[1])
        go_neighbors[g] = [t for t, _ in go_neighbors[g][:args.topk]]
    print(f'  Genes with GO neighbors: {len(go_neighbors)}')

    print('[3/5] Generating descriptions...')
    descriptions = {}
    for g in gene_names:
        if g in go_neighbors and go_neighbors[g]:
            nbs = ', '.join(go_neighbors[g])
            desc = f'{g} is functionally related to {nbs} based on Gene Ontology annotation similarity.'
        else:
            desc = f'{g} is a protein-coding gene involved in cellular processes.'
        descriptions[g] = desc

    with open(f'{args.data_dir}embeddings/gene_descriptions.json', 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=2)
    print(f'  Saved descriptions for {len(descriptions)} genes')

    if not args.embed:
        print('\n[SKIP] Embedding disabled (use --embed to enable)')
        print('Sample descriptions:')
        for g in gene_names[:5]:
            print(f'  {g}: {descriptions[g]}')
        return

    print('[4/5] Loading embedding model...')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            # Try sentence-transformers first (newer envs)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(args.model)
            print(f'  Model: {args.model} (sentence-transformers)')
            texts = [descriptions[g] for g in gene_names]
            embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
        except (ImportError, AttributeError) as e:
            print(f'  sentence-transformers unavailable ({str(e)[:50]}), using transformers directly...')
            # Fallback: use transformers.AutoModel directly
            from transformers import AutoTokenizer, AutoModel
            import torch
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            print(f'  Model: {model_name} (transformers)')
            def embed_batch(texts, batch_size=256):
                all_embs = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    inputs = tokenizer(batch, padding=True, truncation=True,
                                       max_length=128, return_tensors='pt')
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Mean pooling
                        attention_mask = inputs['attention_mask']
                        token_emb = outputs.last_hidden_state
                        mask = attention_mask.unsqueeze(-1).float()
                        mean_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1)
                        all_embs.append(mean_emb.numpy())
                return np.concatenate(all_embs, axis=0)
            texts = [descriptions[g] for g in gene_names]
            embeddings = embed_batch(texts, batch_size=256)

    print(f'[5/5] Saving shape {embeddings.shape}...')
    np.save(f'{args.data_dir}embeddings/gene_biollm_embeddings.npy', embeddings)
    print('  Done!')

if __name__ == '__main__':
    main()
