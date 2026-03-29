"""
Deep learning legal document type classifier (BERT + mean-pooled MLP head).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModel, AutoTokenizer

BACKBONE_NAME = "bert-base-uncased"

DOCUMENT_LABELS: List[str] = [
    "petition",
    "judgment",
    "contract",
    "FIR",
    "affidavit",
    "notice",
    "agreement",
]
ID2LABEL: Dict[int, str] = {i: DOCUMENT_LABELS[i] for i in range(len(DOCUMENT_LABELS))}
LABEL2ID: Dict[str, int] = {name: i for i, name in ID2LABEL.items()}
NUM_LABELS = 7

_DEFAULT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = _DEFAULT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "doc_classifier.pt"

_model: LegalDocumentClassifier | None = None
_tokenizer: Any = None
_device: torch.device | None = None
_infer_max_len: int = 512


def _mean_pool(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pool token embeddings; ignore padded positions."""
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class LegalDocumentClassifier(nn.Module):
    """BERT encoder + mean-pooled vector through BatchNorm MLP head."""

    def __init__(self, backbone_name: str = BACKBONE_NAME) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        if hidden != 768:
            raise ValueError(f"Expected hidden size 768, got {hidden}")
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_LABELS),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = _mean_pool(out.last_hidden_state, attention_mask)
        return self.classifier(pooled)


class LegalDocumentDataset(Dataset):
    """Tokenized document texts and integer labels."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_len: int = 512,
    ) -> None:
        self.texts = list(texts)
        self.labels = [int(y) for y in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length.")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label


def _collate_documents(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    attention_mask = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.stack([b[2] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels


def train_model(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    *,
    val_fraction: float = 0.15,
    batch_size: int = 8,
    max_epochs: int = 20,
    max_len: int = 512,
    seed: int = 42,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    Train LegalDocumentClassifier with CrossEntropyLoss and AdamW (lr=3e-5).
    Saves the best checkpoint (lowest validation loss) to models/doc_classifier.pt.
    """
    texts = list(train_texts)
    labels_list = [int(y) for y in train_labels]
    if len(texts) == 0:
        raise ValueError("No training samples provided.")
    for y in labels_list:
        if y not in ID2LABEL:
            raise ValueError(f"Invalid label id {y}; expected 0..{NUM_LABELS - 1}.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    full_ds = LegalDocumentDataset(texts, labels_list, tokenizer, max_len=max_len)

    n = len(full_ds)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    if n_train < 1:
        n_val = max(1, n // 5)
        n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_documents,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_documents,
    )

    model = LegalDocumentClassifier().to(dev)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    history: List[Dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = {k: v.to(dev) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = {k: v.to(dev) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(dev)
                logits = model(**batch_inputs)
                val_loss += criterion(logits, batch_labels).item()
        val_loss /= max(1, len(val_loader))

        history.append(
            {"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": best_state,
        "backbone_name": BACKBONE_NAME,
        "document_labels": DOCUMENT_LABELS,
        "id2label": ID2LABEL,
        "num_labels": NUM_LABELS,
        "max_len": max_len,
        "training_history": history,
        "best_val_loss": best_val,
    }
    torch.save(payload, ckpt_path)

    global _model, _tokenizer, _device, _infer_max_len
    _model = None
    _tokenizer = None
    _device = None
    _infer_max_len = 512

    return {
        "checkpoint_path": str(ckpt_path.resolve()),
        "best_val_loss": best_val,
        "epochs_ran": len(history),
        "history": history,
    }


def _load_bundle(
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> Tuple[LegalDocumentClassifier, Any, torch.device, int]:
    global _model, _tokenizer, _device, _infer_max_len
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train first with train_model()."
        )
    if _model is not None and _tokenizer is not None and _device == dev:
        return _model, _tokenizer, dev, _infer_max_len

    try:
        ckpt = torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=dev)
    backbone = ckpt.get("backbone_name", BACKBONE_NAME)
    max_len = int(ckpt.get("max_len", 512))
    model = LegalDocumentClassifier(backbone_name=backbone).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tok = AutoTokenizer.from_pretrained(backbone)
    _model, _tokenizer, _device = model, tok, dev
    _infer_max_len = max_len
    return model, tok, dev, max_len


def classify_document(
    text: str,
    *,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    Return predicted document type, max softmax confidence, and all class scores.
    """
    model, tokenizer, dev, max_len = _load_bundle(checkpoint_path, device)
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    batch = {k: v.to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        logits = model(**batch)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
    scores_list = probs.cpu().tolist()
    pred_id = int(torch.argmax(probs).item())
    confidence = float(scores_list[pred_id])
    all_scores = {ID2LABEL[i]: float(scores_list[i]) for i in range(NUM_LABELS)}
    return {
        "document_type": ID2LABEL[pred_id],
        "confidence": confidence,
        "all_scores": all_scores,
    }


if __name__ == "__main__":
    samples = [
        ("The petitioner respectfully submits this petition.", 0),
        ("It is hereby ordered and adjudged as follows.", 1),
        ("This agreement is entered into as of the effective date.", 6),
    ]
    txts = [t for t, _ in samples]
    lbs = [y for _, y in samples]
    out = train_model(txts, lbs, max_epochs=2, batch_size=2, val_fraction=0.34)
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=2))
    print(json.dumps(classify_document(txts[0]), indent=2))
