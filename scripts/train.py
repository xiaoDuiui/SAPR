"""Reproducible training entry point for the ICASSP experiments."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from models import ProcePertdata, scpert  # noqa: E402


VARIANTS = {
    "baseline": dict(use_deg_sparse=False, interaction_lambda=0.0,
                     use_adaptive_fusion=False),
    "saf": dict(use_deg_sparse=True, interaction_lambda=0.0,
                use_adaptive_fusion=False),
    "saf_iar": dict(use_deg_sparse=True, interaction_lambda=0.05,
                    use_adaptive_fusion=False),
    "saf_amf": dict(use_deg_sparse=True, interaction_lambda=0.0,
                    use_adaptive_fusion=True),
    "full": dict(use_deg_sparse=True, interaction_lambda=0.05,
                 use_adaptive_fusion=True),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="norman")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparse-ratio", type=float, default=0.25)
    parser.add_argument("--data-root", type=Path,
                        default=WORKSPACE_ROOT / "data")
    parser.add_argument("--embedding-root", type=Path,
                        default=WORKSPACE_ROOT / "embeddings")
    parser.add_argument("--output-root", type=Path,
                        default=REPO_ROOT / "artifacts" / "experiments")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def align_scgpt_embeddings(pert_data, source_path, output_path):
    """Align source symbols to the model gene order; missing genes are zero."""
    source = np.load(source_path, allow_pickle=True)
    source_names = [str(name).upper() for name in source["gene_names"]]
    source_embeddings = source["embeddings"].astype(np.float32)
    source_index = {name: idx for idx, name in enumerate(source_names)}

    if "gene_name" not in pert_data.adata.var:
        raise ValueError("adata.var['gene_name'] is required for embedding alignment")
    symbols = [str(name).upper() for name in pert_data.adata.var["gene_name"]]
    aligned = np.zeros((len(symbols), source_embeddings.shape[1]), dtype=np.float32)
    matched = 0
    for idx, symbol in enumerate(symbols):
        source_idx = source_index.get(symbol)
        if source_idx is not None:
            aligned[idx] = source_embeddings[source_idx]
            matched += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, aligned)
    return matched, len(symbols)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = args.output_root / args.dataset / args.variant / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pert_data = ProcePertdata.PertData(str(args.data_root))
    pert_data.load(DataName=args.dataset)
    pert_data.prepare_split(split="simulation", seed=args.seed)
    pert_data.get_dataloader(batch_size=args.batch_size,
                             test_batch_size=args.batch_size)

    scgpt_source = args.embedding_root / "gene_embeddingss_full_common.npz"
    aligned_path = run_dir / "aligned_scgpt.npy"
    matched, total = align_scgpt_embeddings(pert_data, scgpt_source, aligned_path)
    print(f"Aligned scGPT embeddings: {matched}/{total}; missing genes use zero vectors")

    model = scpert.scPert(
        pert_data,
        device=device,
        weight_bias_track=False,
        proj_name="sapr_icassp",
        exp_name=f"{args.dataset}_{args.variant}_{args.seed}",
        embedding_path=str(aligned_path),
    )
    variant = VARIANTS[args.variant]
    model.model_initialize(
        hidden_size=64,
        use_deg_sparse=variant["use_deg_sparse"],
        deg_sparse_ratio=args.sparse_ratio,
        deg_calibrated_lambda=0.0,
        interaction_lambda=variant["interaction_lambda"],
        use_adaptive_fusion=variant["use_adaptive_fusion"],
        deg_ratio=0.1,
        kge_embedding_path=str(args.embedding_root /
                               "gene_2_kge_comgcn_final_common.npz"),
        scgpt_embedding_path=str(scgpt_source),
    )

    results = model.train(epochs=args.epochs, lr=args.lr, use_parallel=False)
    payload = {
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "sparse_ratio": args.sparse_ratio,
        "embedding_coverage": {"matched": matched, "total": total},
        "results": results,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")

    if args.save_model:
        model.save_model(str(run_dir / "checkpoint"))
    print(f"Saved metrics to {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
