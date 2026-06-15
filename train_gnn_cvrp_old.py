# train_gnn_cvrp.py

import os
import glob
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import GNNPolicy, TopKPairwiseReranker
from gurobi_utilities import GraphDataset, pad_tensor, log


SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "data_arc_features/samples/cvrp/debug"
VALID_DIR = ""

MODEL_DIR = "models/cvrp_gnn"
MODEL_PATH = os.path.join(MODEL_DIR, "gnn_policy.pt")
LOG_PATH = os.path.join(MODEL_DIR, "train_log.txt")

BATCH_SIZE = 32
LR = 1e-2
MAX_EPOCHS = 30
PATIENCE = 40
TOP_K = [1, 3, 5, 10]
USE_RERANKER = True
RERANK_TOP_K = 20
RERANK_LOSS_WEIGHT = 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_prenorm_initialization(policy, train_loader):
    """
    Runs the PreNormLayer statistic-collection phase.

    This must happen before normal training.
    It passes training batches through the model without gradients.
    Each PreNormLayer collects shift/scale statistics and is then frozen.
    """

    log("Starting PreNorm initialization...", LOG_PATH)

    policy.train()
    policy.pre_train_init()

    prenorm_round = 0

    while True:
        prenorm_round += 1
        found_waiting_layer = False

        for batch in train_loader:
            batch = batch.to(DEVICE)

            still_collecting = policy.pre_train(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )

            if still_collecting:
                found_waiting_layer = True

        updated_layer = policy.pre_train_next()

        if updated_layer is None:
            break

        log(
            f"PreNorm round {prenorm_round}: initialized {updated_layer.__class__.__name__}",
            LOG_PATH,
        )

    log("Finished PreNorm initialization.", LOG_PATH)

def build_topk_reranker_batch(
    logits,
    variable_embeddings,
    variable_features,
    candidates,
    nb_candidates,
    candidate_choices,
    top_k=20,
):
    """
    Builds fixed-size [B, K, ...] tensors for the reranker.

    Important:
    - batch.candidates are already global variable indices because your
      BipartiteMILPData.__inc__ increments them by num_variables.
    - candidate_choices are local positions inside each sample's candidate list.
    """

    B = int(nb_candidates.numel())
    device = logits.device

    topk_global_candidates = []
    topk_base_logits = []
    topk_embeddings = []
    topk_raw_features = []
    topk_labels = []

    start = 0

    for b in range(B):
        n = int(nb_candidates[b].item())
        end = start + n

        cand_global = candidates[start:end]
        cand_logits = logits[cand_global]

        k = min(top_k, n)

        top_local = torch.topk(cand_logits, k=k, dim=0).indices
        top_global = cand_global[top_local]

        # Pad to top_k if n < top_k.
        if k < top_k:
            pad_count = top_k - k

            pad_global = top_global[-1:].repeat(pad_count)
            pad_logits = cand_logits[top_local[-1:]].repeat(pad_count)
            pad_emb = variable_embeddings[top_global[-1:]].repeat(pad_count, 1)
            pad_raw = variable_features[top_global[-1:]].repeat(pad_count, 1)

            top_global_padded = torch.cat([top_global, pad_global], dim=0)
            top_logits_padded = torch.cat([cand_logits[top_local], pad_logits], dim=0)
            top_emb_padded = torch.cat([variable_embeddings[top_global], pad_emb], dim=0)
            top_raw_padded = torch.cat([variable_features[top_global], pad_raw], dim=0)
        else:
            top_global_padded = top_global
            top_logits_padded = cand_logits[top_local]
            top_emb_padded = variable_embeddings[top_global]
            top_raw_padded = variable_features[top_global]

        true_local = int(candidate_choices[b].item())

        # Reranker can only be trained if expert is inside top-K.
        match = (top_local[:k] == true_local).nonzero(as_tuple=False)

        if match.numel() == 0:
            rerank_label = -1
        else:
            rerank_label = int(match[0].item())

        topk_global_candidates.append(top_global_padded)
        topk_base_logits.append(top_logits_padded)
        topk_embeddings.append(top_emb_padded)
        topk_raw_features.append(top_raw_padded)
        topk_labels.append(rerank_label)

        start = end

    return {
        "topk_global_candidates": torch.stack(topk_global_candidates, dim=0),
        "topk_base_logits": torch.stack(topk_base_logits, dim=0),
        "topk_embeddings": torch.stack(topk_embeddings, dim=0),
        "topk_raw_features": torch.stack(topk_raw_features, dim=0),
        "topk_labels": torch.tensor(topk_labels, dtype=torch.long, device=device),
    }


def process(policy, loader, optimizer=None, reranker=None):
    is_train = optimizer is not None

    if is_train:
        policy.train()
        if reranker is not None:
            reranker.train()
    else:
        policy.eval()
        if reranker is not None:
            reranker.eval()

    total_loss = 0.0
    total_base_loss = 0.0
    total_rerank_loss = 0.0

    total_graphs = 0
    total_kacc = np.zeros(len(TOP_K), dtype=np.float64)

    total_candidates = 0
    min_candidates = 10**9
    max_candidates = 0

    total_rank_percentile = 0.0
    total_mrr = 0.0

    total_rerank_seen = 0
    total_rerank_usable = 0
    total_rerank_correct = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(DEVICE)

            logits, variable_embeddings = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                return_embeddings=True,
            )

            cand_logits = logits[batch.candidates]
            cand_logits = pad_tensor(cand_logits, batch.nb_candidates)

            true_choices = batch.candidate_choices.view(-1)

            base_loss = F.cross_entropy(
                cand_logits,
                true_choices,
                reduction="mean",
            )

            loss = base_loss
            rerank_loss_value = torch.tensor(0.0, device=DEVICE)

            # --------------------------------------------------
            # Optional top-K reranker
            # --------------------------------------------------
            if reranker is not None and USE_RERANKER:
                rerank_batch = build_topk_reranker_batch(
                    logits=logits,
                    variable_embeddings=variable_embeddings,
                    variable_features=batch.variable_features,
                    candidates=batch.candidates,
                    nb_candidates=batch.nb_candidates,
                    candidate_choices=true_choices,
                    top_k=RERANK_TOP_K,
                )

                rerank_logits = reranker(
                    cand_emb=rerank_batch["topk_embeddings"],
                    cand_raw=rerank_batch["topk_raw_features"],
                    cand_base_logits=rerank_batch["topk_base_logits"],
                )

                rerank_labels = rerank_batch["topk_labels"]
                usable_mask = rerank_labels >= 0

                total_rerank_seen += int(rerank_labels.numel())
                total_rerank_usable += int(usable_mask.sum().item())

                if usable_mask.any():
                    rerank_loss_value = F.cross_entropy(
                        rerank_logits[usable_mask],
                        rerank_labels[usable_mask],
                        reduction="mean",
                    )

                    loss = base_loss + RERANK_LOSS_WEIGHT * rerank_loss_value

                    rerank_pred = rerank_logits[usable_mask].argmax(dim=-1)
                    total_rerank_correct += int(
                        (rerank_pred == rerank_labels[usable_mask]).sum().item()
                    )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + ([] if reranker is None else list(reranker.parameters())),
                    max_norm=1.0,
                )
                optimizer.step()

            true_scores = pad_tensor(
                batch.candidate_scores,
                batch.nb_candidates,
                pad_value=-1e8,
            )

            true_best = true_scores.max(dim=-1, keepdim=True).values

            kacc = []
            for k in TOP_K:
                kk = min(k, cand_logits.shape[-1])
                pred_topk = cand_logits.topk(kk, dim=-1).indices
                pred_scores = true_scores.gather(-1, pred_topk)
                acc = (pred_scores == true_best).any(dim=-1).float().mean().item()
                kacc.append(acc)

            # --------------------------------------------------
            # Rank percentile + MRR for base GNN
            # --------------------------------------------------
            pred_order = torch.argsort(cand_logits, dim=-1, descending=True)

            batch_rank_percentiles = []
            batch_mrr = []

            for i in range(batch.num_graphs):
                n = int(batch.nb_candidates[i].item())
                true_i = int(true_choices[i].item())

                order_i = pred_order[i, :n]
                rank0 = (order_i == true_i).nonzero(as_tuple=False).item()
                rank1 = rank0 + 1

                batch_rank_percentiles.append(rank1 / n)
                batch_mrr.append(1.0 / rank1)

            num_graphs = batch.num_graphs

            total_loss += loss.item() * num_graphs
            total_base_loss += base_loss.item() * num_graphs
            total_rerank_loss += float(rerank_loss_value.item()) * num_graphs

            total_kacc += np.asarray(kacc) * num_graphs
            total_graphs += num_graphs

            total_candidates += int(batch.nb_candidates.sum().item())
            min_candidates = min(min_candidates, int(batch.nb_candidates.min().item()))
            max_candidates = max(max_candidates, int(batch.nb_candidates.max().item()))

            total_rank_percentile += sum(batch_rank_percentiles)
            total_mrr += sum(batch_mrr)

    avg_loss = total_loss / total_graphs
    avg_base_loss = total_base_loss / total_graphs
    avg_rerank_loss = total_rerank_loss / total_graphs
    avg_kacc = total_kacc / total_graphs

    stats = {
        "avg_candidates": total_candidates / total_graphs,
        "min_candidates": min_candidates,
        "max_candidates": max_candidates,
        "mean_rank_percentile": total_rank_percentile / total_graphs,
        "mean_mrr": total_mrr / total_graphs,
        "base_loss": avg_base_loss,
        "rerank_loss": avg_rerank_loss,
        "rerank_usable_ratio": (
            total_rerank_usable / max(total_rerank_seen, 1)
        ),
        "rerank_acc": (
            total_rerank_correct / max(total_rerank_usable, 1)
        ),
    }

    return avg_loss, avg_kacc, stats


def main():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "sample_*.pkl")))
    valid_files = sorted(glob.glob(os.path.join(VALID_DIR, "sample_*.pkl")))

    if len(valid_files) == 0:
        random.shuffle(train_files)
        split = int(0.8 * len(train_files))
        valid_files = train_files[split:]
        train_files = train_files[:split]

    log(f"Train samples: {len(train_files)}", LOG_PATH)
    log(f"Valid samples: {len(valid_files)}", LOG_PATH)


    

    train_data = GraphDataset(train_files)
    valid_data = GraphDataset(valid_files)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    policy = GNNPolicy(
        var_nfeats=19,
        cons_nfeats=5,
        edge_nfeats=1,
        emb_size=64,
        n_rounds=2,
        dropout=0.05,
    ).to(DEVICE)

    # --------------------------------------------------
    # PreNorm initialization before training
    # --------------------------------------------------
    run_prenorm_initialization(policy, train_loader)

    params = list(policy.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    best_valid_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_kacc, train_stats = process(
            policy,
            train_loader,
            optimizer,
            reranker=None,
        )

        valid_loss, valid_kacc, valid_stats = process(
            policy,
            valid_loader,
            optimizer=None,
            reranker=None,
        )
        

        train_msg = " ".join(
            [f"train_acc@{k}: {acc:.4f}" for k, acc in zip(TOP_K, train_kacc)]
        )
        valid_msg = " ".join(
            [f"valid_acc@{k}: {acc:.4f}" for k, acc in zip(TOP_K, valid_kacc)]
        )

        log(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | valid_loss={valid_loss:.5f} | "
            f"{train_msg} | {valid_msg} | "
            f"train_cands_avg={train_stats['avg_candidates']:.1f} "
            f"train_cands_min={train_stats['min_candidates']} "
            f"train_cands_max={train_stats['max_candidates']} | "
            f"valid_cands_avg={valid_stats['avg_candidates']:.1f} "
            f"valid_cands_min={valid_stats['min_candidates']} "
            f"valid_cands_max={valid_stats['max_candidates']} | "
            f"train_rank_pct={train_stats['mean_rank_percentile']:.4f} "
            f"valid_rank_pct={valid_stats['mean_rank_percentile']:.4f} | "
            f"train_mrr={train_stats['mean_mrr']:.4f} "
            f"valid_mrr={valid_stats['mean_mrr']:.4f}"
            f"train_base_loss={train_stats['base_loss']:.5f} "
            f"valid_base_loss={valid_stats['base_loss']:.5f} | "
            f"train_rerank_loss={train_stats['rerank_loss']:.5f} "
            f"valid_rerank_loss={valid_stats['rerank_loss']:.5f} | "
            f"train_rerank_usable={train_stats['rerank_usable_ratio']:.3f} "
            f"valid_rerank_usable={valid_stats['rerank_usable_ratio']:.3f} | "
            f"train_rerank_acc={train_stats['rerank_acc']:.4f} "
            f"valid_rerank_acc={valid_stats['rerank_acc']:.4f}",
            LOG_PATH,
        )

        

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            bad_epochs = 0
            torch.save(
                {
                    "policy": policy.state_dict(),
                    
                },
                MODEL_PATH,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            log("Early stopping.", LOG_PATH)
            break

    log(f"Best valid loss: {best_valid_loss:.5f}", LOG_PATH)


if __name__ == "__main__":
    main()







    {
            "category_id": 1,
            "name": "C01_very_easy_8_customers_Q035",
            "n_customers": 8,
            "Q_ratio": 0.35,
        },
        {
            "category_id": 2,
            "name": "C02_easy_9_customers_Q035",
            "n_customers": 9,
            "Q_ratio": 0.35,
        },
        {
            "category_id": 3,
            "name": "C03_easy_10_customers_Q030",
            "n_customers": 10,
            "Q_ratio": 0.30,
        },
        {
            "category_id": 4,
            "name": "C04_low_mid_11_customers_Q030",
            "n_customers": 11,
            "Q_ratio": 0.30,
        },
    







    difficulty_categories = [

        {
            "category_id": 1,
            "name": "C01_very_easy_8_customers_Q035",
            "n_customers": 8,
            "Q_ratio": 0.35,
        },
        {
            "category_id": 2,
            "name": "C02_easy_9_customers_Q035",
            "n_customers": 9,
            "Q_ratio": 0.35,
        },
        {
            "category_id": 3,
            "name": "C03_easy_10_customers_Q030",
            "n_customers": 10,
            "Q_ratio": 0.30,
        },
        {
            "category_id": 4,
            "name": "C04_low_mid_11_customers_Q030",
            "n_customers": 11,
            "Q_ratio": 0.30,
        },
        
        {
            "category_id": 5,
            "name": "C05_mid_12_customers_Q030",
            "n_customers": 12,
            "Q_ratio": 0.30,
        },
        {
            "category_id": 6,
            "name": "C06_mid_13_customers_Q025",
            "n_customers": 13,
            "Q_ratio": 0.25,
        },
        {
            "category_id": 7,
            "name": "C07_mid_hard_14_customers_Q025",
            "n_customers": 14,
            "Q_ratio": 0.25,
        },
        {
            "category_id": 8,
            "name": "C08_hard_15_customers_Q025",
            "n_customers": 15,
            "Q_ratio": 0.25,
        },
        { 
            "category_id": 9,
            "name": "C09_harder_16_customers_Q022",
            "n_customers": 16,
            "Q_ratio": 0.22,
        },
        {
            "category_id": 10,
            "name": "C10_hardest_18_customers_Q022",
            "n_customers": 18,
            "Q_ratio": 0.22,
        },
    ]