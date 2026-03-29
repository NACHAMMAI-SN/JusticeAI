"""
Central training pipeline for legal text classifiers (PyTorch + HuggingFace tokenizers).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

_DEFAULT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = _DEFAULT_ROOT / "models"
TRAINING_CURVES_PATH = MODELS_DIR / "training_curves.png"


def _tokenizer_from_model(model: nn.Module) -> Any:
    """Use AutoTokenizer when the model exposes a HuggingFace backbone config."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError(
            "Pass tokenizer= to run_training_pipeline when the model has no .backbone."
        )
    cfg = backbone.config
    name = getattr(cfg, "name_or_path", None) or getattr(cfg, "_name_or_path", None)
    if not name:
        raise ValueError("Could not read backbone name_or_path; pass tokenizer= explicitly.")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 10
    lr: float = 2e-5
    warmup_steps: int = 100
    max_len: int = 512
    train_split: float = 0.8
    patience: int = 3
    seed: int = 42


class LegalTextDataset(Dataset):
    """Tokenized legal texts with integer labels."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_len: int,
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


def _collate_legal_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    attention_mask = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.stack([b[2] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Tuple[float, float]:
    """One training epoch with tqdm; returns average loss and accuracy."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    n_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="train", leave=False)
    for batch_inputs, batch_labels in pbar:
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(**batch_inputs)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().tolist())
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / max(1, n_batches)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return float(avg_loss), float(acc)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_names: List[str] | None = None,
) -> Tuple[float, float, str]:
    """
    Returns accuracy, macro F1 score, and sklearn classification_report string.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch_inputs, batch_labels in dataloader:
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        batch_labels = batch_labels.to(device)
        logits = model(**batch_inputs)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = (
        f1_score(all_labels, all_preds, average="macro", zero_division=0)
        if all_labels
        else 0.0
    )
    labels_present = sorted(set(all_labels) | set(all_preds))
    names = None
    if target_names is not None:
        names = [target_names[i] for i in labels_present if i < len(target_names)]
    report = classification_report(
        all_labels,
        all_preds,
        labels=labels_present,
        target_names=names,
        zero_division=0,
    )
    return float(acc), float(f1), report


def run_training_pipeline(
    model: nn.Module,
    texts: Sequence[str],
    labels: Sequence[int],
    model_save_path: str | Path,
    config: TrainingConfig,
    *,
    tokenizer: Any | None = None,
    target_names: List[str] | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    80/20 train/val split, full training with validation each epoch,
    best checkpoint by validation loss, loss/accuracy plots, final report on val.
    If tokenizer is None, builds one from model.backbone (HuggingFace) when possible.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    tok = tokenizer if tokenizer is not None else _tokenizer_from_model(model)

    torch.manual_seed(config.seed)
    texts_list = list(texts)
    labels_list = [int(y) for y in labels]
    n = len(texts_list)
    if n == 0:
        raise ValueError("No samples provided.")

    g = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(n * config.train_split)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]
    if not train_indices or not val_indices:
        raise ValueError("train_split produced an empty train or validation set.")

    full_ds = LegalTextDataset(texts_list, labels_list, tok, config.max_len)
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate_legal_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=_collate_legal_batch,
    )

    optimizer = AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    steps_per_epoch = max(1, len(train_loader))
    total_steps = config.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "epochs_ran": 0,
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }

    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    patience_left = config.patience

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, dev
        )
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        model.eval()
        with torch.no_grad():
            n_v = len(val_loader)
            all_preds: List[int] = []
            all_true: List[int] = []
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = {k: v.to(dev) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(dev)
                logits = model(**batch_inputs)
                val_loss += loss_fn(logits, batch_labels).item()
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_true.extend(batch_labels.cpu().tolist())
            val_loss /= max(1, n_v)
            val_acc = float(accuracy_score(all_true, all_preds)) if all_true else 0.0
            val_f1 = float(
                f1_score(all_true, all_preds, average="macro", zero_division=0)
            ) if all_true else 0.0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["epochs_ran"] = epoch

        print(
            f"Epoch {epoch}/{config.epochs}  "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_val_loss"] = best_val_loss
            history["best_epoch"] = epoch
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (patience={config.patience}).")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "config": {
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "lr": config.lr,
                "warmup_steps": config.warmup_steps,
                "max_len": config.max_len,
                "train_split": config.train_split,
                "patience": config.patience,
            },
            "best_val_loss": best_val_loss,
            "best_epoch": history["best_epoch"],
        },
        save_path,
    )

    epochs_ran = history["epochs_ran"]
    epochs_axis = list(range(1, epochs_ran + 1))

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax_loss.plot(epochs_axis, history["train_loss"], label="train_loss", marker="o")
    ax_loss.plot(epochs_axis, history["val_loss"], label="val_loss", marker="o")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and validation loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs_axis, history["train_acc"], label="train_acc", marker="o")
    ax_acc.plot(epochs_axis, history["val_acc"], label="val_acc", marker="o")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training and validation accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim(0.0, 1.05)

    plt.savefig(TRAINING_CURVES_PATH, dpi=150)
    plt.close(fig)

    _, _, final_report = evaluate(model, val_loader, dev, target_names=target_names)
    print("\nFinal validation classification report:\n")
    print(final_report)

    history["training_curves_path"] = str(TRAINING_CURVES_PATH.resolve())
    history["model_save_path"] = str(save_path.resolve())
    return history


def generate_sample_legal_data() -> Tuple[List[str], List[int]]:
    """
    60 Indian legal-style queries: 20 per class.
    0 = personal-and-family, 1 = business-consumer-criminal, 2 = consultation.
    """
    personal_family = [
        "My husband has filed for divorce; what are my rights under the Hindu Marriage Act regarding maintenance?",
        "Can I claim permanent alimony if I have no independent income after 15 years of marriage?",
        "How do I obtain a protection order under the Protection of Women from Domestic Violence Act in Delhi?",
        "What is the procedure for mutual consent divorce under Section 13B HMA and how long does it take?",
        "My wife is denying visitation; how can I enforce child custody orders from the family court?",
        "Are gifts received during marriage considered joint property under Indian succession law?",
        "I want to adopt a child; what are the eligibility criteria under the Juvenile Justice Act?",
        "Can a Hindu daughter claim a share in ancestral property after her father's death in 2024?",
        "How is child support calculated when the father works abroad and income is in foreign currency?",
        "What grounds can I use for cruelty under Section 13(1)(ia) HMA based on mental harassment?",
        "Is live-in relationship registration possible in Maharashtra and does it affect maintenance claims?",
        "My in-laws are demanding dowry; how do I file an FIR and what sections apply?",
        "Can mediation be mandatory before contested divorce under the Family Courts Act?",
        "How do I challenge a ex parte maintenance order from the magistrate court?",
        "What documents are needed for registration of a Hindu marriage under Special Marriage Act conversion?",
        "My partner refuses divorce; can I file on grounds of irretrievable breakdown if the Supreme Court allows?",
        "How is guardianship of a minor decided when parents are separated and the child is 10 years old?",
        "Can grandparents seek visitation rights if the mother has sole custody?",
        "What is the limitation period to appeal a family court order on alimony in India?",
        "I face emotional abuse but no physical injury; is that enough for divorce and protection under DV Act?",
    ]

    business_consumer_criminal = [
        "My supplier breached a contract; can I sue for specific performance in a commercial suit in Mumbai?",
        "A cheque I issued bounced; what is the procedure under Section 138 of the Negotiable Instruments Act?",
        "The builder delayed possession by 3 years; can I approach RERA and claim compensation?",
        "I received a GST notice for alleged input tax credit mismatch; how do I reply within 30 days?",
        "My employer terminated me without notice; is this wrongful termination under the Industrial Disputes Act?",
        "I want to file a consumer complaint in NCDRC for a defective car; what is the pecuniary jurisdiction?",
        "Someone forged my signature on a loan document; what IPC sections apply and how do I file an FIR?",
        "Can I be arrested without warrant for a non-bailable offence under the BNSS and how do I apply for bail?",
        "My startup's co-founder diluted my shares; what remedies exist under the Companies Act 2013?",
        "I am accused in a cyber fraud case involving UPI; what should I expect during police investigation?",
        "The competition commission fined our cartel; can we appeal to NCLAT and on what grounds?",
        "A customer filed a criminal defamation complaint over a Google review; is that maintainable?",
        "I imported goods and customs seized the shipment; what is the appeal process before CESTAT?",
        "My tenant defaulted on rent for a commercial shop; can I evict under state rent control laws?",
        "I was named in an FIR for rioting though I was not present; how do I seek quashing in High Court?",
        "An insurance company denied my health claim citing pre-existing disease; how do I fight in consumer court?",
        "I want to file an insolvency petition against a corporate debtor; what is the minimum default amount?",
        "Police want my phone password in a money laundering probe; what are my rights under BNSS and IT Act?",
        "My MSME vendor payment was delayed beyond 45 days; can I file under the MSMED Act interest provisions?",
        "I received a show-cause notice from SEBI for insider trading; what is the timeline to respond?",
    ]

    consultation = [
        "I would like to book a consultation with a family lawyer to discuss custody options next week.",
        "Can your firm schedule a phone consultation regarding a startup funding agreement?",
        "I need a 30-minute consultation to understand if I have a case for workplace harassment.",
        "Please arrange an in-person consultation at your Bangalore office for a property partition matter.",
        "I want to speak to a criminal lawyer for a brief consultation before deciding to file an FIR.",
        "How much do you charge for an initial consultation on a consumer forum complaint?",
        "I am looking for a free first consultation for legal aid eligibility under state schemes.",
        "Can I get a video consultation with a senior advocate for a second opinion on a High Court order?",
        "I would like to schedule a consultation to review a draft employment contract before I sign.",
        "Please confirm availability for a consultation on Saturday for a cheque bounce case.",
        "I need a consultation only—no litigation yet—about trademark infringement against my brand.",
        "Can your team provide a paid consultation for NRI tax residency and Indian property rules?",
        "I want to book a joint consultation with my spouse for mediation on divorce terms.",
        "Please slot a consultation after I share documents by email regarding a landlord dispute.",
        "I am seeking a one-hour consultation on POCSO implications for a school incident.",
        "Can I reschedule my consultation from Tuesday to Thursday for a corporate compliance issue?",
        "I would like a preliminary consultation to assess merits before paying court fees.",
        "Do you offer same-day emergency consultation for anticipatory bail preparation?",
        "I need a consultation with a labour law specialist about gratuity calculation and PF disputes.",
        "Please send a payment link for the consultation fee and calendar invite for legal advice on adoption.",
    ]

    assert len(personal_family) == 20
    assert len(business_consumer_criminal) == 20
    assert len(consultation) == 20

    texts = personal_family + business_consumer_criminal + consultation
    labels = [0] * 20 + [1] * 20 + [2] * 20
    return texts, labels


if __name__ == "__main__":
    from dl_intent_classifier import LegalIntentClassifier

    m = LegalIntentClassifier()
    tx, y = generate_sample_legal_data()
    cfg = TrainingConfig(epochs=3, batch_size=8, warmup_steps=20)
    names = [
        "personal-and-family",
        "business-consumer-criminal",
        "consultation",
    ]
    hist = run_training_pipeline(
        m,
        tx,
        y,
        MODELS_DIR / "pipeline_intent_demo.pt",
        cfg,
        target_names=names,
    )
    print("History keys:", list(hist.keys()))
    print("Curves saved:", hist.get("training_curves_path"))
