"""
Deep learning legal intent classifier (Legal-BERT + MLP head).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

BACKBONE_NAME = "nlpaueb/legal-bert-base-uncased"

ID2LABEL: Dict[int, str] = {
    0: "personal-and-family-legal-assistance",
    1: "business-consumer-and-criminal-legal-assistance",
    2: "consultation",
}
LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = 3

_DEFAULT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = _DEFAULT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "intent_classifier.pt"

# Lazy singletons for inference
_model: LegalIntentClassifier | None = None
_tokenizer: Any = None
_device: torch.device | None = None
_infer_max_length: int = 256


class LegalIntentClassifier(nn.Module):
    """Legal-BERT encoder + MLP classifier on CLS token."""

    def __init__(self, backbone_name: str = BACKBONE_NAME) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        if hidden != 768:
            raise ValueError(f"Expected hidden size 768, got {hidden}")
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_LABELS),
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
        cls_vec = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_vec)


class _IntentDataset(Dataset):
    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        return item, self.labels[idx]


def _ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _collate_batch(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    keys = batch[0][0].keys()
    batched: Dict[str, torch.Tensor] = {
        k: torch.stack([b[0][k] for b in batch], dim=0) for k in keys
    }
    labels = torch.stack([b[1] for b in batch], dim=0)
    return batched, labels


def get_model_summary(model: nn.Module | None = None) -> Dict[str, Any]:
    """
    Return total/trainable parameter counts and per-layer shape info.
    If model is None, builds LegalIntentClassifier (random head; backbone from HF).
    """
    m = model if model is not None else LegalIntentClassifier()
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    layers: List[Dict[str, Any]] = []
    for name, p in m.named_parameters():
        layers.append(
            {
                "name": name,
                "shape": list(p.shape),
                "numel": p.numel(),
                "trainable": p.requires_grad,
            }
        )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "layers": layers,
    }


def train_model(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    *,
    val_fraction: float = 0.15,
    batch_size: int = 16,
    max_epochs: int = 50,
    max_length: int = 256,
    seed: int = 42,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    Train LegalIntentClassifier with AdamW, linear warmup, grad clipping, early stopping.
    Saves best checkpoint (by validation loss) to models/intent_classifier.pt.
    """
    texts = list(train_texts)
    labels_list = [int(y) for y in train_labels]
    if len(texts) != len(labels_list):
        raise ValueError("train_texts and train_labels must have the same length.")
    if len(texts) == 0:
        raise ValueError("No training samples provided.")

    for y in labels_list:
        if y not in ID2LABEL:
            raise ValueError(f"Invalid label id {y}; expected 0, 1, or 2.")

    _ensure_models_dir()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    label_tensor = torch.tensor(labels_list, dtype=torch.long)

    full_ds = _IntentDataset(
        {k: enc[k] for k in enc.keys()},
        label_tensor,
    )

    n = len(full_ds)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    if n_train < 1:
        n_val = max(1, n // 5)
        n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )

    model = LegalIntentClassifier().to(dev)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    steps_per_epoch = max(1, len(train_loader))
    num_training_steps = max_epochs * steps_per_epoch
    warmup_steps = max(1, int(0.1 * num_training_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = nn.CrossEntropyLoss()
    best_val = float("inf")
    patience = 3
    patience_left = patience
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = {k: v.to(dev) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(dev)
                logits = model(**batch_inputs)
                val_loss += criterion(logits, batch_labels).item()
        val_loss /= len(val_loader)

        history.append(
            {"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": best_state,
        "backbone_name": BACKBONE_NAME,
        "id2label": ID2LABEL,
        "num_labels": NUM_LABELS,
        "max_length": max_length,
        "training_history": history,
        "best_val_loss": best_val,
    }
    torch.save(payload, ckpt_path)

    global _model, _tokenizer, _device, _infer_max_length
    _model = None
    _tokenizer = None
    _device = None
    _infer_max_length = 256

    return {
        "checkpoint_path": str(ckpt_path.resolve()),
        "best_val_loss": best_val,
        "epochs_ran": len(history),
        "history": history,
    }


def _load_inference_bundle(
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> Tuple[LegalIntentClassifier, Any, torch.device, int]:
    global _model, _tokenizer, _device, _infer_max_length
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train the model first with train_model()."
        )
    if _model is not None and _tokenizer is not None and _device == dev:
        return _model, _tokenizer, dev, _infer_max_length

    try:
        ckpt = torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=dev)
    backbone = ckpt.get("backbone_name", BACKBONE_NAME)
    max_len = int(ckpt.get("max_length", 256))
    model = LegalIntentClassifier(backbone_name=backbone).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tok = AutoTokenizer.from_pretrained(backbone)
    _model, _tokenizer, _device = model, tok, dev
    _infer_max_length = max_len
    return model, tok, dev, max_len


def predict_intent(
    text: str,
    *,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    Return predicted intent string, confidence (max softmax prob), and all class scores.
    """
    model, tokenizer, dev, max_len = _load_inference_bundle(checkpoint_path, device)
    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
    scores_list = probs.cpu().tolist()
    pred_id = int(torch.argmax(probs).item())
    confidence = float(scores_list[pred_id])
    all_scores = {ID2LABEL[i]: float(scores_list[i]) for i in range(NUM_LABELS)}
    return {
        "intent": ID2LABEL[pred_id],
        "confidence": confidence,
        "all_scores": all_scores,
    }


if __name__ == "__main__":
    demo_texts = [
        "How do I file for divorce and child custody?",
        "My small business received a cease and desist letter.",
        "I would like to book a consultation to discuss my case.",
    ]
    demo_labels = [0, 1, 2]
    print("Summary (untrained backbone + head):")
    print(json.dumps(get_model_summary(), indent=2)[:2000], "...\n")
    out = train_model(demo_texts, demo_labels, max_epochs=5, batch_size=2)
    print("train_model:", json.dumps({k: v for k, v in out.items() if k != "history"}, indent=2))
    print("predict_intent:", json.dumps(predict_intent(demo_texts[0]), indent=2))
