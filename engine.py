from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
ENGINE_NAMES = ["Naive Bayes", "SVM", "LSTM"]
DEFAULT_FALLBACK_THRESHOLD = 0.60
DEFAULT_MARGIN_THRESHOLD = 0.08


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def clean_text(text: str) -> str:
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
    return data


def build_samples(data: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    seen_pairs: set[Tuple[str, str]] = set()

    for intent in data["intents"]:
        tag = str(intent["tag"]).strip()
        for pattern in intent.get("patterns", []):
            cleaned = clean_text(pattern)
            pair = (cleaned, tag)
            if cleaned and pair not in seen_pairs:
                texts.append(cleaned)
                labels.append(tag)
                seen_pairs.add(pair)

    label_names = sorted(set(labels))
    if len(label_names) < 2:
        raise ValueError("Dataset must contain at least two distinct intents.")
    return texts, labels, label_names


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
    return (
        [texts[int(i)] for i in train_idx],
        [texts[int(i)] for i in test_idx],
        [labels[int(i)] for i in train_idx],
        [labels[int(i)] for i in test_idx],
    )


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


@dataclass
class PredictionResult:
    intent: str
    confidence: float
    alternatives: List[Tuple[str, float]]
    is_fallback: bool
    reason: str = ""


@dataclass
class ModelBundle:
    models: Dict[str, Any]
    lstm: "LSTMEngine"
    labels: List[str]
    x_test: List[str]
    y_test: List[str]
    x_train: List[str]
    y_train: List[str]
    metadata: Dict[str, Any]


class Vocabulary:
    def __init__(self, max_size: int = 8000):
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
        ids = [self.word_to_id.get(token, 1) for token in tokenize(text)[:max_len]]
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
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(self.dropout(combined))


class LSTMEngine:
    def __init__(
        self,
        label_names: Sequence[str],
        max_vocab: int = 8000,
        max_len: int = 32,
        epochs: int = 18,
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
        self.history: List[Dict[str, float]] = []

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self.vocab.encode(text, self.max_len) for text in texts], dtype=np.int64)

    def fit(self, x_train: Sequence[str], y_train: Sequence[str]) -> None:
        start = time.perf_counter()
        self.vocab.fit(x_train)
        x = self._encode_texts(x_train)
        y = np.asarray([self.label_to_id[label] for label in y_train], dtype=np.int64)

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

        loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        self.model = LSTMClassifier(len(self.vocab.word_to_id), len(self.label_names))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state: Dict[str, torch.Tensor] | None = None
        patience = 4
        stale = 0
        self.history.clear()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()

                train_loss += float(loss.item()) * len(yb)
                train_correct += int((logits.argmax(dim=1) == yb).sum().item())
                train_total += len(yb)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(x_val)
                val_loss = float(criterion(val_logits, y_val).item())
                val_acc = float((val_logits.argmax(dim=1) == y_val).float().mean().item())

            train_loss /= max(1, train_total)
            train_acc = train_correct / max(1, train_total)
            self.history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                }
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
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
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()
        return probs


class _FeatureFactory:
    @staticmethod
    def build() -> FeatureUnion:
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                        max_df=0.98,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        sublinear_tf=True,
                        min_df=1,
                        strip_accents="unicode",
                    ),
                ),
            ]
        )


def train_all_models(dataset_path: str | Path) -> ModelBundle:
    set_seed()
    data = load_dataset(dataset_path)
    texts, labels, label_names = build_samples(data)
    x_train, x_test, y_train, y_test = make_shared_split(texts, labels)

    nb_start = time.perf_counter()
    nb = Pipeline(
        [
            ("features", _FeatureFactory.build()),
            ("model", MultinomialNB(alpha=0.05)),
        ]
    )
    nb.fit(x_train, y_train)
    nb_seconds = time.perf_counter() - nb_start

    svm_start = time.perf_counter()
    svm = Pipeline(
        [
            ("features", _FeatureFactory.build()),
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
            "Naive Bayes": {"model": nb, "training_seconds": nb_seconds},
            "SVM": {"model": svm, "training_seconds": svm_seconds},
        },
        lstm=lstm,
        labels=label_names,
        x_test=x_test,
        y_test=y_test,
        x_train=x_train,
        y_train=y_train,
        metadata={
            "dataset_path": str(dataset_path),
            "num_samples": len(texts),
            "num_intents": len(label_names),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "seed": SEED,
            "test_size": 0.20,
        },
    )


def probabilities_for(bundle: ModelBundle, engine: str, texts: Sequence[str]) -> np.ndarray:
    cleaned = [clean_text(text) for text in texts]
    if engine == "LSTM":
        return bundle.lstm.predict_proba(cleaned)
    return bundle.models[engine]["model"].predict_proba(cleaned)


def predict_top_k(bundle: ModelBundle, engine: str, text: str, k: int = 3) -> List[Tuple[str, float]]:
    if not clean_text(text):
        return []
    probs = probabilities_for(bundle, engine, [text])[0]
    if engine == "LSTM":
        classes = bundle.lstm.label_names
    else:
        classes = list(bundle.models[engine]["model"].classes_)
    order = np.argsort(probs)[::-1][:k]
    return [(str(classes[int(i)]), float(probs[int(i)])) for i in order]


def predict(
    bundle: ModelBundle,
    engine: str,
    text: str,
    threshold: float = DEFAULT_FALLBACK_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> PredictionResult:
    top = predict_top_k(bundle, engine, text, k=3)
    if not top:
        return PredictionResult("unknown", 0.0, [("unknown", 0.0)], True, "empty_question")

    intent, confidence = top[0]
    second_confidence = top[1][1] if len(top) > 1 else 0.0
    margin = confidence - second_confidence

    reasons: List[str] = []
    fallback = False
    if intent == "unknown":
        fallback = True
        reasons.append("model_predicted_unknown")
    if confidence < threshold:
        fallback = True
        reasons.append(f"confidence_below_{threshold:.2f}")
    if margin < margin_threshold and confidence < 0.85:
        fallback = True
        reasons.append(f"small_top1_top2_margin")

    return PredictionResult(intent, confidence, top, fallback, ", ".join(reasons))


def evaluate_engine(bundle: ModelBundle, engine: str) -> EvaluationResult:
    probs = probabilities_for(bundle, engine, bundle.x_test)
    classes = bundle.lstm.label_names if engine == "LSTM" else list(bundle.models[engine]["model"].classes_)
    predictions = np.asarray([classes[int(i)] for i in np.argmax(probs, axis=1)])

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

    training_seconds = (
        bundle.lstm.training_seconds if engine == "LSTM" else bundle.models[engine]["training_seconds"]
    )
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
        y_true=list(bundle.y_test),
        labels=list(bundle.labels),
        confusion=confusion_matrix(bundle.y_test, predictions, labels=bundle.labels),
        training_seconds=float(training_seconds),
    )


def evaluate_all(bundle: ModelBundle) -> Dict[str, EvaluationResult]:
    return {engine: evaluate_engine(bundle, engine) for engine in ENGINE_NAMES}


def evaluate_challenge_set(
    bundle: ModelBundle,
    challenge_cases: Sequence[Dict[str, str]],
    engine: str,
    threshold: float = DEFAULT_FALLBACK_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> Dict[str, Any]:
    y_true: List[str] = []
    y_pred: List[str] = []
    fallback_count = 0
    rows: List[Dict[str, Any]] = []

    for case in challenge_cases:
        question = case["question"]
        expected = case["intent"]
        result = predict(bundle, engine, question, threshold, margin_threshold)
        predicted = "unknown" if result.is_fallback else result.intent
        y_true.append(expected)
        y_pred.append(predicted)
        fallback_count += int(result.is_fallback)
        rows.append(
            {
                "Question": question,
                "Expected": expected,
                "Predicted": predicted,
                "Confidence": result.confidence,
                "Correct": expected == predicted,
                "Fallback": result.is_fallback,
                "Reason": result.reason,
            }
        )

    labels = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fallback_count": fallback_count,
        "total": len(challenge_cases),
        "rows": rows,
    }


def dataset_statistics(data: Dict[str, Any]) -> Dict[str, Any]:
    patterns = sum(len(i.get("patterns", [])) for i in data["intents"])
    responses = sum(len(i.get("responses", [])) for i in data["intents"])
    counts = {i["tag"]: len(i.get("patterns", [])) for i in data["intents"]}

    normalized_to_tags: Dict[str, set[str]] = {}
    for item in data["intents"]:
        for p in item.get("patterns", []):
            normalized = clean_text(p)
            normalized_to_tags.setdefault(normalized, set()).add(item["tag"])
    conflicting = sum(1 for tags in normalized_to_tags.values() if len(tags) > 1)
    duplicate_exact = sum(max(0, len(tags) - 1) for tags in normalized_to_tags.values() if len(tags) > 1)

    section_names = [
        "class_timetable",
        "exam_schedule",
        "operating_hours",
        "events",
        "department_contacts",
        "campus_places",
        "course_fees",
    ]

    return {
        "intents": len(data["intents"]),
        "patterns": patterns,
        "responses": responses,
        "avg_patterns_per_intent": patterns / max(1, len(data["intents"])),
        "largest_intent": max(counts, key=counts.get),
        "largest_intent_count": counts[max(counts, key=counts.get)],
        "smallest_intent": min(counts, key=counts.get),
        "smallest_intent_count": counts[min(counts, key=counts.get)],
        "pattern_conflicts": conflicting,
        "pattern_conflict_extras": duplicate_exact,
        "structured_sections": {key: len(data.get(key, [])) for key in section_names if key in data},
        "intent_counts": counts,
    }


def _tokens_for_matching(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "is", "are", "where", "what", "when", "how", "can", "could",
        "please", "tell", "me", "do", "i", "my", "to", "of", "for", "and", "in", "on", "at",
        "from", "with", "about", "which", "does", "there", "you", "your", "university", "campus",
    }
    return {token for token in tokenize(text) if token not in stop}


def _find_all_matches(
    query: str,
    items: Sequence[Dict[str, Any]],
    fields: Sequence[str],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    q_tokens = _tokens_for_matching(query)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    q_clean = clean_text(query)

    for item in items:
        item_text = " ".join(str(item.get(field, "")) for field in fields)
        item_clean = clean_text(item_text)
        item_tokens = _tokens_for_matching(item_text)
        overlap = len(q_tokens & item_tokens)
        score = overlap / max(1.0, len(q_tokens))
        if item_clean and item_clean in q_clean:
            score += 1.5
        for field in fields[:1]:
            field_value = clean_text(str(item.get(field, "")))
            if field_value and field_value in q_clean:
                score += 2.0
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def extract_entities(data: Dict[str, Any], query: str) -> Dict[str, str]:
    q = clean_text(query)
    entities: Dict[str, str] = {}

    # Level entity for fee questions.
    for level in ["foundation", "diploma", "degree"]:
        if level in q:
            entities["level"] = level.title()
            break

    # Course / programme / place / department exact-ish phrase matching.
    for item in data.get("course_fees", []):
        programme = str(item.get("Programme", ""))
        if programme and clean_text(programme) in q:
            entities["programme"] = programme
            entities["level"] = str(item.get("Level", entities.get("level", "")))
            break

    for item in data.get("class_timetable", []) + data.get("exam_schedule", []):
        course = str(item.get("Course", ""))
        if course and clean_text(course) in q:
            entities["course"] = course
            break
        code = course.split()[0] if course else ""
        if code and code.lower() in q:
            entities["course"] = course
            break

    for item in data.get("department_contacts", []):
        department = str(item.get("Department", ""))
        if department and clean_text(department) in q:
            entities["department"] = department
            break

    for item in data.get("campus_places", []):
        place = str(item.get("Place", ""))
        if place and clean_text(place) in q:
            entities["place"] = place
            break

    # Helpful short-code/entity patterns.
    course_code = re.search(r"\b[a-z]{3}\d{4}\b", q)
    if course_code:
        entities["course_code"] = course_code.group(0).upper()

    gates = re.search(r"\bgate\s*([1-4])\b", q)
    if gates:
        entities["gate"] = f"Gate {gates.group(1)}"

    return {key: value for key, value in entities.items() if value}



def decide_visual_response(intent: str, query: str = "") -> Dict[str, Any]:
    """Decide whether an intent should use an image, table, contact card, or text."""
    if intent in {"library_location", "campus_location", "parking", "hostel", "campus_dining"}:
        key_map = {
            "library_location": "library_map",
            "campus_location": "campus_map",
            "parking": "parking_map",
            "hostel": "hostel_map",
            "campus_dining": "dining_map",
        }
        return {
            "needs_image": True,
            "type": "pdf",
            "visual_key": "campus_map_pdf",
            "title": "Campus Location Map",
            "reason": "This intent is location-oriented, so the provided campus map PDF can complement the text answer.",
        }
    if intent in {"course_fee_inquiry", "timetable", "exam", "office_hours"}:
        return {
            "needs_image": False,
            "type": "table",
            "visual_key": None,
            "title": "Structured data response",
            "reason": "Structured information is clearer as a table than as an image.",
        }
    if intent == "department_contact":
        return {
            "needs_image": False,
            "type": "contact",
            "visual_key": None,
            "title": "Contact information",
            "reason": "Contact details are clearer in a contact card.",
        }
    return {
        "needs_image": False,
        "type": "text",
        "visual_key": None,
        "title": "Text response",
        "reason": "A text response is sufficient for this intent.",
    }

def dynamic_response(data: Dict[str, Any], intent: str, query: str) -> str | None:
    entities = extract_entities(data, query)

    if intent == "course_fee_inquiry" and data.get("course_fees"):
        items = data["course_fees"]
        matches: List[Dict[str, Any]] = []
        if entities.get("programme"):
            matches = [
                item for item in items
                if clean_text(str(item.get("Programme", ""))) == clean_text(entities["programme"])
            ]
        elif entities.get("level"):
            matches = [
                item for item in items
                if str(item.get("Level", "")).lower() == entities["level"].lower()
            ]
        else:
            matches = _find_all_matches(
                query,
                items,
                ["Level", "Programme", "Estimated_Fee_Malaysian", "Estimated_Fee_International"],
                limit=3,
            )

        if not matches:
            return "I can help with the available estimated course fees. Try asking about Foundation, Diploma, or Degree, or name the programme."

        lines = ["Here are the estimated course fees matching your question:"]
        for item in matches[:5]:
            lines.append(
                f"• {item['Programme']} ({item['Duration']}): "
                f"Malaysian {item['Estimated_Fee_Malaysian']}; "
                f"International {item['Estimated_Fee_International']}."
            )
        lines.append("These are estimates; verify the latest official fee schedule before payment.")
        return "\n".join(lines)

    if intent == "tuition_fee":
        return (
            "Tuition fees can normally be paid through the university payment portal or approved payment methods. "
            "Please check your student account for the exact amount, balance, payment deadline and any refund conditions."
        )

    if intent == "timetable" and data.get("class_timetable"):
        matches = _find_all_matches(query, data["class_timetable"], ["Day", "Time", "Course", "Venue"], limit=6)
        if entities.get("course"):
            matches = [m for m in data["class_timetable"] if clean_text(str(m.get("Course", ""))) == clean_text(entities["course"])]
        elif entities.get("course_code"):
            matches = [m for m in data["class_timetable"] if entities["course_code"].lower() in clean_text(str(m.get("Course", "")))]
        if not matches:
            matches = data["class_timetable"][:6]
        lines = ["Class timetable entries:"]
        for item in matches[:6]:
            lines.append(f"• {item['Day']} {item['Time']} — {item['Course']} at {item['Venue']}.")
        return "\n".join(lines)

    if intent == "exam" and data.get("exam_schedule"):
        matches = _find_all_matches(query, data["exam_schedule"], ["Date", "Time", "Course", "Venue"], limit=6)
        if entities.get("course"):
            matches = [m for m in data["exam_schedule"] if clean_text(str(m.get("Course", ""))) == clean_text(entities["course"])]
        elif entities.get("course_code"):
            matches = [m for m in data["exam_schedule"] if entities["course_code"].lower() in clean_text(str(m.get("Course", "")))]
        if not matches:
            matches = data["exam_schedule"]
        lines = ["Exam schedule entries:"]
        for item in matches[:6]:
            lines.append(f"• {item['Date']} {item['Time']} — {item['Course']} at {item['Venue']}.")
        return "\n".join(lines)

    if intent in {"library_hours", "office_hours"} and data.get("operating_hours"):
        if intent == "library_hours":
            candidates = [
                item for item in data["operating_hours"]
                if "library" in clean_text(str(item.get("Service", "")))
            ]
        else:
            candidates = _find_all_matches(
                query,
                data["operating_hours"],
                ["Service", "Monday-Friday", "Weekend / public holiday", "Phone"],
                limit=5,
            )
        if not candidates:
            candidates = data["operating_hours"][:5]
        lines = ["Service operating hours:"]
        for item in candidates[:5]:
            lines.append(
                f"• {item['Service']}: Monday–Friday {item.get('Monday-Friday', 'Not listed')}; "
                f"weekend/public holiday {item.get('Weekend / public holiday', 'Not listed')}."
            )
        return "\n".join(lines)

    if intent == "department_contact" and data.get("department_contacts"):
        matches: List[Dict[str, Any]]
        if entities.get("department"):
            matches = [
                item for item in data["department_contacts"]
                if clean_text(str(item.get("Department", ""))) == clean_text(entities["department"])
            ]
        else:
            matches = _find_all_matches(
                query,
                data["department_contacts"],
                ["Department", "Help with", "Location", "Phone", "Contact"],
                limit=3,
            )
        if not matches:
            matches = data["department_contacts"][:3]
        lines = ["Relevant department contact(s):"]
        for item in matches[:3]:
            lines.append(
                f"• {item['Department']} — {item.get('Help with', '')}. "
                f"Phone: {item.get('Phone', 'N/A')}. Email: {item.get('Contact', 'N/A')}. "
                f"Location: {item.get('Location', 'N/A')}."
            )
        return "\n".join(lines)

    if intent == "campus_events" and data.get("events"):
        lines = ["Upcoming campus events in the supplied dataset:"]
        for item in data["events"]:
            lines.append(f"• {item['Date']} — {item['Event']} at {item['Location']} ({item['Audience']}).")
        return "\n".join(lines)

    if intent in {"campus_location", "library_location", "campus_dining", "parking", "hostel", "student_services"} and data.get("campus_places"):
        fields = ["Place", "Category", "Location", "Notes", "Directions"]
        if entities.get("place"):
            matches = [
                item for item in data["campus_places"]
                if clean_text(str(item.get("Place", ""))) == clean_text(entities["place"])
            ]
        else:
            matches = _find_all_matches(query, data["campus_places"], fields, limit=3)

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
            matches = data["campus_places"][:3]
        lines = ["Campus information:"]
        for item in matches[:3]:
            lines.append(f"• {item['Place']} — {item['Location']}. {item.get('Directions', '')}")
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
        for item in data.get("intents", []):
            if item.get("tag") == intent:
                choices = item.get("responses", [])
                break
    if not choices:
        return "Sorry, I could not find a prepared answer for that topic. Please contact the relevant university office."

    digest = __import__("hashlib").sha256(clean_text(query).encode("utf-8")).hexdigest()
    return choices[int(digest[:12], 16) % len(choices)]


def average_inference_ms(bundle: ModelBundle, engine: str, sample_texts: Sequence[str], repeats: int = 10) -> float:
    samples = list(sample_texts)[: min(10, len(sample_texts))]
    if not samples:
        return 0.0
    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        probabilities_for(bundle, engine, samples)
    elapsed = time.perf_counter() - start
    calls = max(1, repeats) * len(samples)
    return elapsed * 1000 / calls


@dataclass
class DataQualityReport:
    total_patterns: int
    total_intents: int
    empty_patterns: int
    exact_duplicates: int
    conflicting_patterns: int
    min_patterns: int
    max_patterns: int
    mean_patterns: float
    intents_under_20: List[str] = field(default_factory=list)


def build_quality_report(data: Dict[str, Any]) -> DataQualityReport:
    counts = {i["tag"]: len(i.get("patterns", [])) for i in data["intents"]}
    normalized: Dict[str, set[str]] = {}
    empty = 0
    for item in data["intents"]:
        for pattern in item.get("patterns", []):
            cleaned = clean_text(pattern)
            if not cleaned:
                empty += 1
            normalized.setdefault(cleaned, set()).add(item["tag"])
    exact_duplicates = sum(max(0, len(tags) - 1) for tags in normalized.values())
    conflicting = sum(1 for tags in normalized.values() if len(tags) > 1)
    return DataQualityReport(
        total_patterns=sum(counts.values()),
        total_intents=len(counts),
        empty_patterns=empty,
        exact_duplicates=exact_duplicates,
        conflicting_patterns=conflicting,
        min_patterns=min(counts.values()),
        max_patterns=max(counts.values()),
        mean_patterns=float(np.mean(list(counts.values()))),
        intents_under_20=sorted([tag for tag, count in counts.items() if count < 20]),
    )
