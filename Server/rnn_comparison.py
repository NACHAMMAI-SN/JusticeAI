"""
Standalone RNN vs BERT comparison on LEDGAR (top-3 labels).
Run from Server/:  python rnn_comparison.py

Does not import or modify any other project modules.
Outputs only to models/rnn_* (does not touch intent_classifier.pt).
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Third-party (pip install datasets torch matplotlib)
from datasets import load_dataset

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SEED = 42
SAMPLES_PER_CLASS = 200
TRAIN_FRAC = 0.85
MAX_LEN = 128
EMBED_DIM = 128
HIDDEN = 128
BATCH_SIZE = 32
EPOCHS = 20
EARLY_PATIENCE = 5
MIN_WORD_FREQ = 2
PAD_IDX = 0
UNK_IDX = 1

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BERT_RESULTS = {
    "Legal-BERT (AdamW)": {"val_accuracy": 1.00, "final_val_loss": 0.02},
    "DistilBERT (AdamW)": {"val_accuracy": 1.00, "final_val_loss": 0.03},
    "BERT-base (AdamW)": {"val_accuracy": 1.00, "final_val_loss": 0.02},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ledgar_subset():
    """Top-3 labels by frequency; 200 samples per class; map labels to 0,1,2."""
    ds = load_dataset("lex_glue", "ledgar")
    train_split = ds["train"]

    # Resolve text column
    cols = train_split.column_names
    text_key = "text" if "text" in cols else ("text_a" if "text_a" in cols else cols[0])
    label_key = "label" if "label" in cols else cols[1]

    labels = train_split[label_key]
    # Normalize label to hashable str for counting
    norm = [str(x) for x in labels]
    counts = Counter(norm)
    top3 = [lab for lab, _ in counts.most_common(3)]
    if len(top3) < 3:
        raise RuntimeError("Need at least 3 distinct labels in LEDGAR train split.")

    lab_to_new = {top3[i]: i for i in range(3)}

    # Collect indices per class
    by_class = {0: [], 1: [], 2: []}
    for i, row in enumerate(train_split):
        t = str(row[label_key])
        if t not in lab_to_new:
            continue
        by_class[lab_to_new[t]].append(i)

    rng = random.Random(SEED)
    selected = []
    for c in (0, 1, 2):
        idxs = by_class[c][:]
        rng.shuffle(idxs)
        take = min(SAMPLES_PER_CLASS, len(idxs))
        if take < SAMPLES_PER_CLASS:
            print(
                f"Warning: class {c} has only {len(idxs)} samples; using {take}."
            )
        selected.extend(idxs[:take])

    rng.shuffle(selected)
    texts = []
    y = []
    for i in selected:
        row = train_split[i]
        texts.append(str(row[text_key]))
        t = str(row[label_key])
        y.append(lab_to_new[t])

    # 85/15 train/val
    n = len(texts)
    n_train = int(n * TRAIN_FRAC)
    idx = list(range(n))
    rng.shuffle(idx)
    tr_idx = set(idx[:n_train])
    train_texts = [texts[i] for i in range(n) if i in tr_idx]
    train_y = [y[i] for i in range(n) if i in tr_idx]
    val_texts = [texts[i] for i in range(n) if i not in tr_idx]
    val_y = [y[i] for i in range(n) if i not in tr_idx]

    return train_texts, train_y, val_texts, val_y


def tokenize_whitespace(text: str) -> list[str]:
    return text.lower().split()


def build_vocab(train_texts: list[str], min_freq: int) -> dict[str, int]:
    wc: Counter[str] = Counter()
    for t in train_texts:
        wc.update(tokenize_whitespace(t))
    word2idx = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    idx = 2
    for w, c in wc.items():
        if c >= min_freq:
            word2idx[w] = idx
            idx += 1
    return word2idx


def encode(texts: list[str], word2idx: dict[str, int]) -> torch.Tensor:
    out = torch.zeros(len(texts), MAX_LEN, dtype=torch.long)
    for i, text in enumerate(texts):
        ids = [word2idx.get(w, UNK_IDX) for w in tokenize_whitespace(text)]
        ids = ids[:MAX_LEN]
        if ids:
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


class SeqDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def lengths_from_batch(x: torch.Tensor) -> torch.Tensor:
    return (x != PAD_IDX).sum(dim=1).clamp(min=1)


class ClassifierHead(nn.Module):
    """hidden -> 64 -> 3 with ReLU + Dropout(0.3)"""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, h):
        return self.net(h)


class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(EMBED_DIM, HIDDEN, num_layers=1, batch_first=True)
        self.head = ClassifierHead(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lens = lengths_from_batch(x)
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed)
        feat = h_n[-1]
        return self.head(feat)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN, num_layers=1, batch_first=True)
        self.head = ClassifierHead(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lens = lengths_from_batch(x)
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        feat = h_n[-1]
        return self.head(feat)


class StackedLSTMModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            EMBED_DIM, HIDDEN, num_layers=3, dropout=0.3, batch_first=True
        )
        self.head = ClassifierHead(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lens = lengths_from_batch(x)
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        feat = h_n[-1]
        return self.head(feat)


class BidirectionalLSTMModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            EMBED_DIM, HIDDEN, num_layers=1, bidirectional=True, batch_first=True
        )
        self.head = ClassifierHead(HIDDEN * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lens = lengths_from_batch(x)
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # 1 layer bi: h_n[0] forward, h_n[1] backward
        feat = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.head(feat)


MODEL_REGISTRY = [
    ("SimpleRNN", "simple_rnn", SimpleRNNModel),
    ("LSTM", "lstm", LSTMModel),
    ("StackedLSTM", "stacked_lstm", StackedLSTMModel),
    ("BidirectionalLSTM", "bi_lstm", BidirectionalLSTMModel),
]


def make_optimizer(name: str, params):
    if name == "Adam":
        return optim.Adam(params, lr=0.001)
    if name == "AdamW":
        return optim.AdamW(params, lr=0.001, weight_decay=0.01)
    if name == "SGD":
        return optim.SGD(params, lr=0.01, momentum=0.9)
    raise ValueError(name)


def train_one_run(
    display_name: str,
    file_slug_model: str,
    file_slug_opt: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_name: str,
) -> dict:
    model = model.to(DEVICE)
    opt = make_optimizer(optimizer_name, model.parameters())
    crit = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_state = None
    best_val_acc_at_best = 0.0
    patience = EARLY_PATIENCE
    epochs_trained = 0

    ckpt_path = MODELS_DIR / f"rnn_{file_slug_model}_{file_slug_opt}.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        model.eval()
        v_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                v_loss_sum += crit(logits, yb).item()
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_loss = v_loss_sum / max(1, len(val_loader))
        val_acc = correct / max(1, total)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"[{display_name} | {optimizer_name}] Epoch {epoch}/{EPOCHS} — "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {100 * val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_acc_at_best = val_acc
            patience = EARLY_PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print(
                    f"Early stopping at epoch {epoch} for {display_name} + {optimizer_name}"
                )
                epochs_trained = epoch
                break
        epochs_trained = epoch

    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "display_name": display_name,
                "optimizer": optimizer_name,
                "best_val_loss": best_val_loss,
                "val_accuracy_at_best_loss": best_val_acc_at_best,
            },
            ckpt_path,
        )
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    best_val_accuracy = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0
    min_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")

    return {
        "model_name": display_name,
        "optimizer": optimizer_name,
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_loss": float(min_val_loss),
        "epochs_trained": int(epochs_trained),
        "history": history,
        "checkpoint": str(ckpt_path),
    }


def main():
    set_seed(SEED)
    print("Loading LEDGAR (lex_glue / ledgar)...")
    train_texts, train_y, val_texts, val_y = load_ledgar_subset()
    print(
        f"Train: {len(train_texts)}  Val: {len(val_texts)}  "
        f"(top-3 labels, {SAMPLES_PER_CLASS}/class target)"
    )

    word2idx = build_vocab(train_texts, MIN_WORD_FREQ)
    vocab_size = len(word2idx)
    print(f"Vocabulary size (incl. PAD/UNK): {vocab_size}")

    x_train = encode(train_texts, word2idx)
    y_train = torch.tensor(train_y, dtype=torch.long)
    x_val = encode(val_texts, word2idx)
    y_val = torch.tensor(val_y, dtype=torch.long)

    train_ds = SeqDataset(x_train, y_train)
    val_ds = SeqDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizers = ["Adam", "AdamW", "SGD"]
    opt_file = {"Adam": "adam", "AdamW": "adamw", "SGD": "sgd"}

    all_results: list[dict] = []

    t0 = time.time()
    for disp, slug, Cls in MODEL_REGISTRY:
        for oname in optimizers:
            set_seed(SEED)
            model = Cls(vocab_size)
            res = train_one_run(
                disp, slug, opt_file[oname], model, train_loader, val_loader, oname
            )
            all_results.append(res)

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed / 60:.1f} min")

    # Save JSON
    json_path = MODELS_DIR / "rnn_all_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # --- Plots: color per model, linestyle per optimizer ---
    colors = {
        "simple_rnn": "#1f77b4",
        "lstm": "#ff7f0e",
        "stacked_lstm": "#2ca02c",
        "bi_lstm": "#d62728",
    }
    linestyles = {"Adam": "-", "AdamW": "--", "SGD": "-."}

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    for res in all_results:
        disp = res["model_name"]
        opt = res["optimizer"]
        slug = next(s for d, s, _ in MODEL_REGISTRY if d == disp)
        h = res["history"]["val_accuracy"]
        vl = res["history"]["val_loss"]
        ep = range(1, len(h) + 1)
        label = f"{disp} - {opt}"
        ax1.plot(
            ep,
            h,
            label=label,
            color=colors[slug],
            linestyle=linestyles[opt],
            linewidth=1.8,
        )
        ax2.plot(
            ep,
            vl,
            label=label,
            color=colors[slug],
            linestyle=linestyles[opt],
            linewidth=1.8,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title("Validation Accuracy per Epoch — All RNN Variants & Optimizers")
    ax1.legend(fontsize=7, loc="lower right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    fig1.tight_layout()
    fig1.savefig(MODELS_DIR / "rnn_accuracy_per_epoch.png", dpi=150)
    plt.close(fig1)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss per Epoch — All RNN Variants & Optimizers")
    ax2.legend(fontsize=7, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(MODELS_DIR / "rnn_loss_per_epoch.png", dpi=150)
    plt.close(fig2)

    # Grouped bar: 4 models, 3 optimizers
    model_order = [m[1] for m in MODEL_REGISTRY]
    model_labels = [m[0] for m in MODEL_REGISTRY]
    n_models = len(model_labels)
    x_centers = list(range(n_models))
    width = 0.25
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for i, oname in enumerate(optimizers):
        accs = []
        for slug in model_order:
            r = next(
                (
                    z
                    for z in all_results
                    if next(s for d, s, _ in MODEL_REGISTRY if d == z["model_name"])
                    == slug
                    and z["optimizer"] == oname
                ),
                None,
            )
            accs.append(r["best_val_accuracy"] if r else 0.0)
        offsets = [xc + (i - 1) * width for xc in x_centers]
        ax3.bar(offsets, accs, width, label=oname)
    ax3.set_xticks(x_centers)
    ax3.set_xticklabels(model_labels, rotation=15, ha="right")
    ax3.set_ylabel("Best validation accuracy")
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Best Validation Accuracy by Model and Optimizer")
    ax3.legend()
    ax3.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(MODELS_DIR / "rnn_best_accuracy_comparison.png", dpi=150)
    plt.close(fig3)

    # Horizontal bar: 12 RNN + 3 BERT
    combined = []
    for r in all_results:
        combined.append(
            {
                "name": f"{r['model_name']} — {r['optimizer']}",
                "acc": r["best_val_accuracy"],
                "kind": "rnn",
            }
        )
    for name, br in BERT_RESULTS.items():
        combined.append({"name": name, "acc": br["val_accuracy"], "kind": "bert"})
    combined.sort(key=lambda z: z["acc"], reverse=True)

    fig4, ax4 = plt.subplots(figsize=(10, 10))
    names = [c["name"] for c in combined]
    accs = [c["acc"] for c in combined]
    colors_b = ["#2E86AB" if c["kind"] == "rnn" else "#28A745" for c in combined]
    y_pos = range(len(names))
    ax4.barh(list(y_pos), accs, color=colors_b)
    ax4.set_yticks(list(y_pos))
    ax4.set_yticklabels(names, fontsize=8)
    ax4.invert_yaxis()
    ax4.set_xlabel("Best validation accuracy")
    ax4.set_xlim(0, 1.05)
    ax4.set_title("Final Model Comparison — RNN vs BERT")
    ax4.legend(
        handles=[
            Patch(facecolor="#2E86AB", label="RNN variants"),
            Patch(facecolor="#28A745", label="BERT (reference)"),
        ],
        loc="lower right",
    )
    ax4.grid(True, axis="x", alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(MODELS_DIR / "final_model_comparison.png", dpi=150)
    plt.close(fig4)

    # Summary table
    rows = sorted(all_results, key=lambda z: z["best_val_accuracy"], reverse=True)
    lines = []
    lines.append(
        f"{'Model':<28} | {'Optimizer':<10} | {'Best Val Acc':>14} | {'Best Val Loss':>14} | {'Epochs':>8}"
    )
    lines.append("-" * 100)
    for r in rows:
        lines.append(
            f"{r['model_name']:<28} | {r['optimizer']:<10} | "
            f"{r['best_val_accuracy']:>14.4f} | {r['best_val_loss']:>14.4f} | {r['epochs_trained']:>8d}"
        )
    best = rows[0]
    lines.append("")
    lines.append(
        f"BEST (highest validation accuracy): {best['model_name']} + {best['optimizer']} "
        f"— Val Accuracy: {100 * best['best_val_accuracy']:.2f}%"
    )
    table_text = "\n".join(lines)
    print("\n" + table_text)
    with open(MODELS_DIR / "rnn_summary_table.txt", "w", encoding="utf-8") as f:
        f.write(table_text + "\n")

    print(
        f"\nBEST MODEL: {best['model_name']} with {best['optimizer']} — "
        f"Val Accuracy: {100 * best['best_val_accuracy']:.2f}%"
    )
    print(f"\nSaved: {json_path}, plots, checkpoints under {MODELS_DIR}")


if __name__ == "__main__":
    main()
