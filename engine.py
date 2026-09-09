
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
ENGINE_NAMES = ["Naive Bayes", "SVM", "LSTM"]


def set_seed(seed: int = SEED) -> None:
    """Make model training as reproducible as reasonably possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    """Normalize text while preserving useful alphanumeric tokens."""
    text = str(text).lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", clean_text(text))


def load_dataset(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "intents" not in data or not data["intents"]:
        raise ValueError("dataset.json must contain a non-empty 'intents' list.")

    # Keep the original structured data so dynamic answers can use it.
    return data


def build_samples(data: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for intent in data["intents"]:
        tag = str(intent["tag"])
        for pattern in intent.get("patterns", []):
            p = clean_text(pattern)
            if p:
                texts.append(p)
                labels.append(tag)

    labels_sorted = sorted(set(labels))
    if len(labels_sorted) < 2:
        raise ValueError("Dataset must contain at least two distinct intents.")
    return texts, labels, labels_sorted


def make_shared_split(
    texts: Sequence[str],
    labels: Sequence[str],
    test_size: float = 0.20,
    seed: int = SEED,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    indices = np.arange(len(texts))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    x_train = [texts[int(i)] for i in train_idx]
    x_test = [texts[int(i)] for i in test_idx]
    y_train = [labels[int(i)] for i in train_idx]
    y_test = [labels[int(i)] for i in test_idx]
    return x_train, x_test, y_train, y_test


@dataclass
class EvaluationResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    predictions: np.ndarray
    y_true: List[str]
    labels: List[str]
    confusion: np.ndarray
    training_seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "predictions": self.predictions,
            "y_true": self.y_true,
            "labels": self.labels,
            "confusion": self.confusion,
            "training_seconds": self.training_seconds,
        }


class Vocabulary:
    def __init__(self, max_size: int = 5000) -> None:
        self.max_size = max_size
        self.word_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts: Sequence[str]) -> None:
        counts: Dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for word, _ in ordered[: max(0, self.max_size - 2)]:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.word_to_id.get(t, 1) for t in tokenize(text)[:max_len]]
        if len(ids) < max_len:
            ids.extend([0] * (max_len - len(ids)))
        return ids


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 96,
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Last forward + last backward hidden state.
        h_forward = hidden[-2]
        h_backward = hidden[-1]
        combined = torch.cat([h_forward, h_backward], dim=1)
        return self.classifier(self.dropout(combined))


class LSTMEngine:
    def __init__(
        self,
        label_names: Sequence[str],
        max_vocab: int = 5000,
        max_len: int = 32,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.002,
    ) -> None:
        self.label_names = list(label_names)
        self.label_to_id = {label: i for i, label in enumerate(self.label_names)}
        self.vocab = Vocabulary(max_size=max_vocab)
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model: LSTMClassifier | None = None
        self.training_seconds = 0.0

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [self.vocab.encode(text, self.max_len) for text in texts],
            dtype=np.int64,
        )

    def fit(self, x_train: Sequence[str], y_train: Sequence[str]) -> None:
        start = time.perf_counter()
        self.vocab.fit(x_train)

        x = self._encode_texts(x_train)
        y = np.asarray([self.label_to_id[yv] for yv in y_train], dtype=np.int64)

        train_idx, val_idx = train_test_split(
            np.arange(len(x)),
            test_size=0.15,
            random_state=SEED,
            stratify=y,
        )
        x_tr = torch.tensor(x[train_idx], dtype=torch.long)
        y_tr = torch.tensor(y[train_idx], dtype=torch.long)
        x_val = torch.tensor(x[val_idx], dtype=torch.long)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model = LSTMClassifier(
            vocab_size=len(self.vocab.word_to_id),
            num_classes=len(self.label_names),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state: Dict[str, torch.Tensor] | None = None
        patience = 3
        stale_epochs = 0

        for _epoch in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(x_val)
                val_loss = float(criterion(val_logits, y_val).item())

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.training_seconds = time.perf_counter() - start

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTM model has not been trained.")
        x = torch.tensor(self._encode_texts(texts), dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


@dataclass
class ModelBundle:
    models: Dict[str, Any]
    lstm: LSTMEngine
    labels: List[str]
    x_test: List[str]
    y_test: List[str]
    metadata: Dict[str, Any]


def train_all_models(dataset_path: str | Path) -> ModelBundle:
    set_seed()
    data = load_dataset(dataset_path)
    texts, labels, label_names = build_samples(data)
    x_train, x_test, y_train, y_test = make_shared_split(texts, labels)

    nb_start = time.perf_counter()
    nb = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                min_df=1,
                                max_df=0.98,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                    ]
                ),
            ),
            ("model", MultinomialNB(alpha=0.05)),
        ]
    )
    nb.fit(x_train, y_train)
    nb_seconds = time.perf_counter() - nb_start

    svm_start = time.perf_counter()
    svm = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                min_df=1,
                                max_df=0.98,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "model",
                SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    svm.fit(x_train, y_train)
    svm_seconds = time.perf_counter() - svm_start

    lstm = LSTMEngine(label_names=label_names)
    lstm.fit(x_train, y_train)

    return ModelBundle(
        models={
            "Naive Bayes": {
                "model": nb,
                "training_seconds": nb_seconds,
            },
            "SVM": {
                "model": svm,
                "training_seconds": svm_seconds,
            },
        },
        lstm=lstm,
        labels=label_names,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "dataset_path": str(dataset_path),
            "num_samples": len(texts),
            "num_intents": len(label_names),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
        },
    )


def probabilities_for(bundle: ModelBundle, engine: str, texts: Sequence[str]) -> np.ndarray:
    if engine == "LSTM":
        return bundle.lstm.predict_proba(texts)
    model = bundle.models[engine]["model"]
    return model.predict_proba(list(texts))


def predict_top_k(
    bundle: ModelBundle,
    engine: str,
    text: str,
    k: int = 3,
) -> List[Tuple[str, float]]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    probs = probabilities_for(bundle, engine, [cleaned])[0]
    order = np.argsort(probs)[::-1][:k]
    # LSTM/SVC probabilities are aligned with their own class order.
    if engine == "LSTM":
        classes = bundle.lstm.label_names
    else:
        classes = list(bundle.models[engine]["model"].classes_)
    return [(str(classes[int(i)]), float(probs[int(i)])) for i in order]


def predict(
    bundle: ModelBundle,
    engine: str,
    text: str,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    top = predict_top_k(bundle, engine, text, k=3)
    if not top:
        return "unknown", 0.0, [("unknown", 0.0)]
    return top[0][0], top[0][1], top


def evaluate_engine(bundle: ModelBundle, engine: str) -> EvaluationResult:
    start = time.perf_counter()
    probs = probabilities_for(bundle, engine, bundle.x_test)
    if engine == "LSTM":
        classes = bundle.lstm.label_names
    else:
        classes = list(bundle.models[engine]["model"].classes_)
    pred_ids = np.argmax(probs, axis=1)
    predictions = np.asarray([classes[i] for i in pred_ids])

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        bundle.y_test,
        predictions,
        labels=bundle.labels,
        average="weighted",
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        bundle.y_test,
        predictions,
        labels=bundle.labels,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(bundle.y_test, predictions, labels=bundle.labels)

    training_seconds = float(
        bundle.lstm.training_seconds
        if engine == "LSTM"
        else bundle.models[engine]["training_seconds"]
    )

    # Keep the result deterministic; evaluation time is not a model metric.
    _ = time.perf_counter() - start

    return EvaluationResult(
        model=engine,
        accuracy=float(accuracy_score(bundle.y_test, predictions)),
        precision=float(weighted_p),
        recall=float(weighted_r),
        f1=float(weighted_f1),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        predictions=predictions,
        y_true=bundle.y_test,
        labels=bundle.labels,
        confusion=cm,
        training_seconds=training_seconds,
    )


def evaluate_all(bundle: ModelBundle) -> Dict[str, EvaluationResult]:
    return {engine: evaluate_engine(bundle, engine) for engine in ENGINE_NAMES}


def dataset_statistics(data: Dict[str, Any]) -> Dict[str, Any]:
    patterns = sum(len(i.get("patterns", [])) for i in data["intents"])
    responses = sum(len(i.get("responses", [])) for i in data["intents"])
    counts = {i["tag"]: len(i.get("patterns", [])) for i in data["intents"]}
    return {
        "intents": len(data["intents"]),
        "patterns": patterns,
        "responses": responses,
        "avg_patterns_per_intent": patterns / max(1, len(data["intents"])),
        "largest_intent": max(counts, key=counts.get),
        "largest_intent_count": counts[max(counts, key=counts.get)],
        "structured_sections": {
            key: len(data.get(key, []))
            for key in [
                "class_timetable",
                "exam_schedule",
                "operating_hours",
                "events",
                "department_contacts",
                "campus_places",
                "course_fees",
            ]
            if key in data
        },
    }


def _tokens_for_matching(text: str) -> set[str]:
    # Remove common conversational glue words from matching.
    stop = {
        "the", "a", "an", "is", "are", "where", "what", "when", "how", "can",
        "could", "please", "tell", "me", "do", "i", "my", "to", "of", "for",
        "and", "in", "on", "at", "from", "with", "about", "which", "does",
        "there", "you", "your", "university", "campus"
    }
    return {t for t in tokenize(text) if t not in stop}


def best_structured_match(query: str, items: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any] | None:
    q_tokens = _tokens_for_matching(query)
    if not q_tokens:
        return None

    best_item = None
    best_score = 0.0
    for item in items:
        item_text = " ".join(str(item.get(f, "")) for f in fields)
        item_tokens = _tokens_for_matching(item_text)
        if not item_tokens:
            continue
        overlap = len(q_tokens & item_tokens)
        exact_bonus = 0.8 if any(
            q_tokens and qt in clean_text(str(item.get(fields[0], ""))).split()
            for qt in q_tokens
        ) else 0.0
        score = overlap / max(1.0, len(q_tokens)) + exact_bonus
        if score > best_score:
            best_score = score
            best_item = item
    return best_item if best_score >= 0.35 else None


def _find_all_matches(query: str, items: Sequence[Dict[str, Any]], fields: Sequence[str], limit: int = 5) -> List[Dict[str, Any]]:
    q_tokens = _tokens_for_matching(query)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in items:
        text = " ".join(str(item.get(f, "")) for f in fields)
        tokens = _tokens_for_matching(text)
        overlap = len(q_tokens & tokens)
        score = overlap / max(1.0, len(q_tokens))
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:limit]]


def dynamic_response(data: Dict[str, Any], intent: str, query: str) -> str | None:
    """Return a data-backed answer for intents that have structured university records."""
    if intent == "course_fee_inquiry" and data.get("course_fees"):
        matches = _find_all_matches(
            query,
            data["course_fees"],
            ["Level", "Programme", "Estimated_Fee_Malaysian", "Estimated_Fee_International"],
            limit=3,
        )
        if not matches:
            return (
                "I can help with the available estimated course fees. "
                "Try asking about Foundation, Diploma, or Degree."
            )
        lines = ["Here are the estimated fees matching your question:"]
        for item in matches:
            lines.append(
                f"• {item['Programme']} ({item['Duration']}): "
                f"Malaysian {item['Estimated_Fee_Malaysian']}; "
                f"International {item['Estimated_Fee_International']}."
            )
        lines.append("These are estimates; please verify the latest official fee schedule before payment.")
        return "\n".join(lines)

    if intent == "timetable" and data.get("class_timetable"):
        matches = _find_all_matches(
            query,
            data["class_timetable"],
            ["Day", "Time", "Course", "Venue"],
            limit=6,
        )
        if not matches:
            matches = data["class_timetable"][:6]
        lines = ["Class timetable entries:"]
        for item in matches:
            lines.append(
                f"• {item['Day']} {item['Time']} — {item['Course']} at {item['Venue']}."
            )
        return "\n".join(lines)

    if intent == "exam" and data.get("exam_schedule"):
        matches = _find_all_matches(
            query,
            data["exam_schedule"],
            ["Date", "Time", "Course", "Venue"],
            limit=6,
        )
        if not matches:
            matches = data["exam_schedule"]
        lines = ["Exam schedule entries:"]
        for item in matches:
            lines.append(
                f"• {item['Date']} {item['Time']} — {item['Course']} at {item['Venue']}."
            )
        return "\n".join(lines)

    if intent in {"library_hours", "office_hours"} and data.get("operating_hours"):
        q = "library" if intent == "library_hours" else query
        matches = _find_all_matches(
            q,
            data["operating_hours"],
            ["Service", "Monday-Friday", "Weekend / public holiday", "Phone"],
            limit=5,
        )
        if not matches:
            matches = data["operating_hours"][:5]
        lines = ["Service operating hours:"]
        for item in matches:
            weekday = item.get("Monday-Friday", "Not listed")
            weekend = item.get("Weekend / public holiday", "Not listed")
            lines.append(
                f"• {item['Service']}: Monday–Friday {weekday}; weekend/public holiday {weekend}."
            )
        return "\n".join(lines)

    if intent == "department_contact" and data.get("department_contacts"):
        matches = _find_all_matches(
            query,
            data["department_contacts"],
            ["Department", "Help with", "Location", "Phone", "Contact"],
            limit=3,
        )
        if not matches:
            matches = data["department_contacts"][:3]
        lines = ["Relevant department contacts:"]
        for item in matches:
            lines.append(
                f"• {item['Department']} — {item.get('Help with', '')}. "
                f"Phone: {item.get('Phone', 'N/A')}. Email: {item.get('Contact', 'N/A')}. "
                f"Location: {item.get('Location', 'N/A')}."
            )
        return "\n".join(lines)

    if intent == "campus_events" and data.get("events"):
        lines = ["Upcoming campus events in the supplied dataset:"]
        for item in data["events"]:
            lines.append(
                f"• {item['Date']} — {item['Event']} at {item['Location']} ({item['Audience']})."
            )
        return "\n".join(lines)

    if intent in {
        "campus_location",
        "library_location",
        "campus_dining",
        "parking",
        "hostel",
        "student_services",
    } and data.get("campus_places"):
        fields = ["Place", "Category", "Location", "Notes", "Directions"]
        matches = _find_all_matches(query, data["campus_places"], fields, limit=3)

        # Narrow category where the intent suggests one.
        category_map = {
            "library_location": {"Library"},
            "campus_dining": {"Dining"},
            "parking": {"Parking"},
            "hostel": {"Accommodation"},
        }
        allowed = category_map.get(intent)
        if allowed:
            cat_matches = [m for m in matches if m.get("Category") in allowed]
            if cat_matches:
                matches = cat_matches

        if not matches:
            if intent == "library_location":
                # Exact library record is always useful.
                matches = [m for m in data["campus_places"] if m.get("Place", "").lower() == "library"][:1]
            else:
                matches = data["campus_places"][:3]

        lines = ["Campus information:"]
        for item in matches[:3]:
            lines.append(
                f"• {item['Place']} — {item['Location']}. {item.get('Directions', '')}"
            )
        return "\n".join(lines)

    return None


def response_for_intent(
    data: Dict[str, Any],
    responses: Dict[str, List[str]],
    intent: str,
    query: str,
) -> str:
    dynamic = dynamic_response(data, intent, query)
    if dynamic:
        return dynamic

    choices = responses.get(intent, [])
    if not choices:
        return (
            "Sorry, I could not find a prepared answer for that topic. "
            "Please contact the relevant university office."
        )

    # Deterministic response selection makes the demo repeatable.
    digest = __import__('hashlib').sha256(clean_text(query).encode('utf-8')).hexdigest()
    idx = int(digest[:12], 16) % len(choices)
    return choices[idx]
