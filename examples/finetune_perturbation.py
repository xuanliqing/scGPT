import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics
from gears import PertData

# -----------------------------------------------------------------------------
# 1. 配置参数 (Configuration)
# -----------------------------------------------------------------------------
DATA_DIR = "./data"
SAVE_DIR = "./save/finetuned_adamson"
PRETRAINED_DIR = "./pretrain"
LOAD_PARAM_PREFIXES = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
SCHEDULE_INTERVAL = 1
EARLY_STOP = 5

DATA_NAME = "adamson"
SPLIT = "simulation"
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
MAX_SEQ_LEN = 1536
INCLUDE_ZERO_GENE = "all"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
set_seed(42)

# -----------------------------------------------------------------------------
# 2. 数据加载与处理
# -----------------------------------------------------------------------------

def load_data_and_vocab():
    print(f"Loading {DATA_NAME} dataset...")
    pert_data = PertData(DATA_DIR)
    pert_data.load(data_name=DATA_NAME)
    pert_data.prepare_split(split=SPLIT, seed=1)
    pert_data.get_dataloader(batch_size=BATCH_SIZE, test_batch_size=EVAL_BATCH_SIZE)
    loaders = pert_data.dataloader

    vocab_file = Path(PRETRAINED_DIR) / "vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Pretrained vocab not found at {vocab_file}")

    vocab = GeneVocab.from_file(vocab_file)
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[PAD_TOKEN])

    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab[PAD_TOKEN] for gene in genes],
        dtype=int,
    )

    return pert_data, loaders, vocab, gene_ids, len(genes)


# -----------------------------------------------------------------------------
# 3. 模型初始化
# -----------------------------------------------------------------------------

def build_model(vocab):
    config_file = Path(PRETRAINED_DIR) / "args.json"
    model_file = Path(PRETRAINED_DIR) / "best_model.pt"

    with open(config_file, "r") as f:
        cfg = json.load(f)

    pretrained_state = torch.load(model_file, map_location=DEVICE)
    use_fast_transformer = cfg.get("use_fast_transformer", True)
    if any("self_attn.Wqkv" in key for key in pretrained_state):
        use_fast_transformer = True
    elif any("self_attn.in_proj_weight" in key for key in pretrained_state):
        use_fast_transformer = False

    model = TransformerGenerator(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        nlayers_cls=cfg.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=cfg.get("dropout", 0.0),
        pad_token=PAD_TOKEN,
        pad_value=0,
        pert_pad_id=cfg.get("pert_pad_id", 0),
        use_fast_transformer=use_fast_transformer,
    )

    print(f"Loading pretrained weights from {model_file}...")
    model_dict = model.state_dict()
    pretrained_dict = dict(pretrained_state)
    if (
        "flag_encoder.weight" in pretrained_dict
        and "pert_encoder.weight" not in pretrained_dict
    ):
        pretrained_dict["pert_encoder.weight"] = pretrained_dict.pop(
            "flag_encoder.weight"
        )
    if LOAD_PARAM_PREFIXES:
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if any(k.startswith(prefix) for prefix in LOAD_PARAM_PREFIXES)
        }
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    load_info = model.load_state_dict(pretrained_dict, strict=False)
    if load_info.missing_keys or load_info.unexpected_keys:
        print(
            "Loaded with missing keys: "
            f"{len(load_info.missing_keys)}, unexpected keys: "
            f"{len(load_info.unexpected_keys)}"
        )

    return model.to(DEVICE)


# -----------------------------------------------------------------------------
# 4. 训练与评估函数
# -----------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, scaler, gene_ids, n_genes):
    model.train()
    total_loss = 0.0

    for batch_data in train_loader:
        batch_data.to(DEVICE)
        batch_size = len(batch_data.y)
        x = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y

        if INCLUDE_ZERO_GENE in ["all", "batch-wise"]:
            if INCLUDE_ZERO_GENE == "all":
                input_gene_ids = torch.arange(n_genes, device=DEVICE)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )

            if len(input_gene_ids) > MAX_SEQ_LEN:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=DEVICE)[
                    :MAX_SEQ_LEN
                ]

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=DEVICE
            )
        else:
            raise ValueError("INCLUDE_ZERO_GENE must be 'all' or 'batch-wise'.")

        with torch.cuda.amp.autocast(enabled=AMP):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            output_values = output_dict["mlm_output"]
            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            loss = masked_mse_loss(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def eval_perturb(loader, model, gene_ids):
    model.eval()
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []

    for batch in loader:
        batch.to(DEVICE)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            predictions = model.pred_perturb(
                batch, include_zero_gene=INCLUDE_ZERO_GENE, gene_ids=gene_ids, amp=AMP
            )
            t = batch.y
            pred.extend(predictions.cpu())
            truth.extend(t.cpu())

            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(predictions[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    pred = torch.stack(pred)
    truth = torch.stack(truth)
    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)

    results = {
        "pert_cat": np.array(pert_cat),
        "pred": pred.detach().cpu().numpy().astype(float),
        "truth": truth.detach().cpu().numpy().astype(float),
        "pred_de": pred_de.detach().cpu().numpy().astype(float),
        "truth_de": truth_de.detach().cpu().numpy().astype(float),
    }
    return results


# -----------------------------------------------------------------------------
# 5. 主流程
# -----------------------------------------------------------------------------

def main():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    pert_data, loaders, vocab, gene_ids, n_genes = load_data_and_vocab()
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]

    model = build_model(vocab)

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, SCHEDULE_INTERVAL, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best_val_corr = -1.0
    patience = 0

    print(f"Start fine-tuning for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, gene_ids, n_genes
        )

        val_results = eval_perturb(val_loader, model, gene_ids)
        val_metrics = compute_perturbation_metrics(val_results, ctrl_adata)
        val_corr = val_metrics["pearson_delta"]

        print(
            f"Epoch {epoch:02d} | Time: {time.time() - start_time:.1f}s | "
            f"Train Loss (MSE): {train_loss:.4f} | "
            f"Val Pearson Delta: {val_corr:.4f}"
        )

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), Path(SAVE_DIR) / "best_model.pt")
            print(f"  >>> New best model saved! (Pearson: {best_val_corr:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print("Early stopping triggered.")
                break

        scheduler.step()

    print(f"Fine-tuning completed. Best Pearson: {best_val_corr:.4f}")
    print(f"Model saved to: {SAVE_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
