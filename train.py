from main import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

lexicon = load_lexicon()
corpus = load_corpus(DATASETS_DIR / "chinese_safe.jsonl")

filtered = []
for r in corpus:
    if r["label"] == "不违规":
        filtered.append((r["text"], 0))
    else:
        subject = r.get("subject", "")
        if subject in TARGET_SUBJECTS:
            filtered.append((r["text"], 1))

texts = [t for t, _ in filtered]
labels = [l for _, l in filtered]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

model = CatoModel()
model.train(X_train, y_train, lexicon)
model.save()

preds, probs = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")
fp = sum(1 for p, t in zip(preds, y_test) if p == 1 and t == 0)
fn = sum(1 for p, t in zip(preds, y_test) if p == 0 and t == 1)
print(f"Accuracy: {acc:.4f}")
print(f"F1 macro: {f1:.4f}")
print(f"FP: {fp}, FN: {fn}")
print(f"Error rate: {(fp + fn) / len(y_test):.4f}")

if acc < 0.90:
    raise SystemExit(f"Accuracy {acc:.4f} < 0.90, training failed")
