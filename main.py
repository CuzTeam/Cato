"""
Cato v4 - 敏感词检测
类别: 色情/反动/政治/脏话侮辱/道德伦理/身体伤害/偏见歧视
核心: LR + TF-IDF(词级+字符级) + 词表特征 + 文本统计特征 + 正式度特征
"""

import json
import pickle
import re
from pathlib import Path

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack

DATASETS_DIR = Path("datasets")
WORDS_DIR = Path("words_repo/Vocabulary")
MODEL_DIR = DATASETS_DIR / "model_v4"
MODEL_DIR.mkdir(exist_ok=True)

TARGET_CATEGORIES = {
    "色情": ["色情类型.txt", "色情词库.txt"],
    "反动": ["反动词库.txt", "GFW补充词库.txt", "新思想启蒙.txt"],
    "政治": ["政治类型.txt", "COVID-19词库.txt", "贪腐词库.txt", "民生词库.txt"],
    "暴恐": ["暴恐词库.txt", "涉枪涉爆.txt"],
    "脏话侮辱": ["其他词库.txt", "补充词库.txt"],
}

CATEGORY_LIST = list(TARGET_CATEGORIES.keys())

TARGET_SUBJECTS = {
    "淫秽色情", "政治错误", "变体词",
    "脏话侮辱", "道德伦理", "身体伤害", "偏见歧视",
}

EXTRA_FEATURE_SCALE = 10.0
PREDICT_THRESHOLD = 0.60


def load_lexicon(categories: dict | None = None) -> dict[str, str]:
    cats = categories or TARGET_CATEGORIES
    lexicon = {}
    for category, filenames in cats.items():
        for fname in filenames:
            fpath = WORDS_DIR / fname
            if not fpath.exists():
                continue
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = line.strip()
                    if w and not w.startswith("#") and len(w) >= 2:
                        lexicon[w] = category
    print(f"  词表总词数: {len(lexicon)}")
    return lexicon


def load_corpus(path: Path) -> list[dict]:
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    return corpus


def jieba_tokenize(text: str) -> str:
    tokens = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return " ".join(tokens + bigrams)


def compute_lexicon_features(texts: list[str], lexicon: dict[str, str]) -> np.ndarray:
    n_cats = len(CATEGORY_LIST)
    n_feats = n_cats + 8
    features = np.zeros((len(texts), n_feats), dtype=np.float64)
    cat_idx = {c: i for i, c in enumerate(CATEGORY_LIST)}

    for i, text in enumerate(texts):
        hit_count = 0
        cats_hit = set()
        for word, cat in lexicon.items():
            if word in text:
                if cat in cat_idx:
                    features[i, cat_idx[cat]] += 1
                    cats_hit.add(cat)
                hit_count += 1

        total_chars = len(text) if text else 1
        features[i, n_cats + 0] = hit_count
        features[i, n_cats + 1] = hit_count / total_chars
        features[i, n_cats + 2] = total_chars
        features[i, n_cats + 3] = np.log1p(len(text))
        features[i, n_cats + 4] = 1.0 if hit_count > 0 else 0.0
        features[i, n_cats + 5] = len(cats_hit)
        if hit_count > 0:
            cat_counts = [features[i, cat_idx[c]] for c in cats_hit]
            features[i, n_cats + 6] = max(cat_counts) / total_chars
        else:
            features[i, n_cats + 6] = 0.0
        for c in cats_hit:
            features[i, n_cats + 7] += (features[i, cat_idx[c]] / hit_count) ** 2

    return features


def compute_text_features(texts: list[str]) -> np.ndarray:
    features = np.zeros((len(texts), 14), dtype=np.float64)
    for i, text in enumerate(texts):
        char_count = len(text)
        if char_count == 0:
            continue
        features[i, 0] = char_count
        punct_count = len(re.findall(r'[，。！？、；：""''…—·\.\!\?\,\;\:]', text))
        features[i, 1] = punct_count / char_count
        special_count = len(re.findall(r'[*#@$%&※★☆◆◇▲△▼▽]', text))
        features[i, 2] = special_count / char_count
        features[i, 3] = len(re.findall(r'[\u4e00-\u9fff]', text)) / char_count
        features[i, 4] = len(re.findall(r'[a-zA-Z0-9]', text)) / char_count
        emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
        features[i, 5] = emoji_count / char_count
        features[i, 6] = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text)) / char_count
        repeat_count = len(re.findall(r'(.)\1{2,}', text))
        features[i, 7] = repeat_count
        excl_count = text.count('！') + text.count('!')
        features[i, 8] = excl_count
        question_count = text.count('？') + text.count('?')
        features[i, 9] = question_count
        hashtag_count = len(re.findall(r'#\S+', text))
        features[i, 10] = hashtag_count
        url_count = len(re.findall(r'https?://\S+', text))
        features[i, 11] = url_count
        features[i, 12] = 1.0 if char_count < 10 else 0.0
        features[i, 13] = len(set(text)) / char_count
    return features


def compute_formality_features(texts: list[str]) -> np.ndarray:
    features = np.zeros((len(texts), 8), dtype=np.float64)
    for i, text in enumerate(texts):
        features[i, 0] = len(re.findall(r'\d{4}年|\d{1,2}月\d{1,2}日|\d{1,2}月', text))
        features[i, 1] = len(re.findall(r'\d+\.?\d*%|百分之', text))
        features[i, 2] = len(re.findall(r'记者|报道|新华社|中新网|央视|人民日报|澎湃|观察者', text))
        formal_punct = len(re.findall(r'[。；：]', text))
        informal_punct = len(re.findall(r'[！？…~]', text))
        features[i, 3] = formal_punct - informal_punct
        sentences = re.split(r'[。！？\!\?]', text)
        sentences = [s for s in sentences if len(s) > 0]
        if sentences:
            features[i, 4] = np.mean([len(s) for s in sentences])
        features[i, 5] = len(re.findall(r'[""「」『』]', text))
        features[i, 6] = len(re.findall(r'据悉|据了解|数据显示|专家表示|相关负责人|通知要求', text))
        features[i, 7] = len(re.findall(r'公司|企业|市场|经济|行业|产业|投资|融资', text))
    return features


class CatoModel:
    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(
            max_features=25000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            max_features=15000,
            analyzer="char",
            ngram_range=(2, 4),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
        )
        self.lr = LogisticRegression(
            C=10.0,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
        )
        self.lexicon = {}
        self.scaler_mean = None
        self.scaler_std = None

    def _build_features(self, texts: list[str], fit: bool = False):
        tokenized = [jieba_tokenize(t) for t in texts]
        if fit:
            word_tfidf = self.word_vectorizer.fit_transform(tokenized)
            char_tfidf = self.char_vectorizer.fit_transform(texts)
        else:
            word_tfidf = self.word_vectorizer.transform(tokenized)
            char_tfidf = self.char_vectorizer.transform(texts)

        lexicon_feats = compute_lexicon_features(texts, self.lexicon)
        text_feats = compute_text_features(texts)
        formality_feats = compute_formality_features(texts)

        extra_feats = np.hstack([lexicon_feats, text_feats, formality_feats])
        if fit:
            self.scaler_mean = extra_feats.mean(axis=0)
            self.scaler_std = extra_feats.std(axis=0)
            self.scaler_std[self.scaler_std == 0] = 1.0
        extra_feats = (extra_feats - self.scaler_mean) / self.scaler_std
        extra_feats *= EXTRA_FEATURE_SCALE

        X = hstack([word_tfidf, char_tfidf, csr_matrix(extra_feats)])
        return X

    def train(self, texts: list[str], labels: list[int], lexicon: dict[str, str]):
        self.lexicon = lexicon
        print("  构建特征...")
        X = self._build_features(texts, fit=True)

        print(f"  特征维度: {X.shape[1]}")
        print("  训练 LR...")
        self.lr.fit(X, labels)
        lr_acc = self.lr.score(X, labels)
        print(f"  LR 训练准确率: {lr_acc:.4f}")

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X = self._build_features(texts, fit=False)
        probs = self.lr.predict_proba(X)[:, 1]
        preds = (probs >= PREDICT_THRESHOLD).astype(int)
        return preds, probs

    def score_text(self, text: str) -> dict:
        preds, probs = self.predict([text])
        p = probs[0]
        if p >= PREDICT_THRESHOLD:
            verdict = "违规"
        elif p >= 0.35:
            verdict = "疑似"
        else:
            verdict = "正常"

        tokens = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]
        notable = []
        for w in tokens:
            if w in self.lexicon:
                notable.append({"word": w, "category": self.lexicon[w]})

        return {"score": round(float(p), 4), "verdict": verdict, "notable_words": notable[:8]}

    def save(self):
        with open(MODEL_DIR / "lr.pkl", "wb") as f:
            pickle.dump(self.lr, f)
        with open(MODEL_DIR / "word_vec.pkl", "wb") as f:
            pickle.dump(self.word_vectorizer, f)
        with open(MODEL_DIR / "char_vec.pkl", "wb") as f:
            pickle.dump(self.char_vectorizer, f)
        with open(MODEL_DIR / "extras.pkl", "wb") as f:
            pickle.dump({
                "lexicon": self.lexicon,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
            }, f)

    def load(self) -> bool:
        needed = ["lr.pkl", "word_vec.pkl", "char_vec.pkl", "extras.pkl"]
        if not all((MODEL_DIR / n).exists() for n in needed):
            return False
        with open(MODEL_DIR / "lr.pkl", "rb") as f:
            self.lr = pickle.load(f)
        with open(MODEL_DIR / "word_vec.pkl", "rb") as f:
            self.word_vectorizer = pickle.load(f)
        with open(MODEL_DIR / "char_vec.pkl", "rb") as f:
            self.char_vectorizer = pickle.load(f)
        with open(MODEL_DIR / "extras.pkl", "rb") as f:
            data = pickle.load(f)
            self.lexicon = data["lexicon"]
            self.scaler_mean = data["scaler_mean"]
            self.scaler_std = data["scaler_std"]
        return True


def evaluate(model: CatoModel, texts: list[str], labels: list[int], dataset_name: str = "测试集"):
    preds, probs = model.predict(texts)
    print(f"\n{'='*55}")
    print(f" {dataset_name} 评估结果")
    print(f"{'='*55}")
    print(classification_report(labels, preds, target_names=["正常", "违规"], digits=4))

    cm = confusion_matrix(labels, preds)
    print(f"  混淆矩阵:")
    print(f"    预测→  正常  违规")
    print(f"    正常  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"    违规  {cm[1][0]:5d}  {cm[1][1]:5d}")

    errors = []
    for text, label, pred, prob in zip(texts, labels, preds, probs):
        if label != pred:
            errors.append({
                "text": text[:100],
                "true": "违规" if label == 1 else "正常",
                "pred": "违规" if pred == 1 else "正常",
                "prob": round(float(prob), 4),
            })

    if errors:
        print(f"\n  错误样本 ({len(errors)} 条):")
        for e in errors[:30]:
            print(f"    真实={e['true']} 预测={e['pred']} p={e['prob']:.4f} | {e['text']}")

    return len(errors), errors


def run_challenge(model: CatoModel, path: Path):
    corpus = load_corpus(path)
    texts = [r["text"] for r in corpus]
    preds, probs = model.predict(texts)

    print(f"\n{'='*55}")
    print(f" Challenge 预测结果 ({path.name}, {len(corpus)} 条)")
    print(f"{'='*55}")

    for r, pred, prob in zip(corpus, preds, probs):
        verdict = "违规" if pred == 1 else "正常"
        icon = "X" if pred == 1 else "O"
        text_preview = r["text"][:60].replace("\n", " ")
        print(f"  {icon} [{verdict}] p={prob:.4f} | {text_preview}")


def main():
    print("=" * 55)
    print("Cato v4 - 敏感词检测")
    print("=" * 55)

    print("\n[1] 加载词表...")
    lexicon = load_lexicon()

    print("\n[2] 加载语料...")
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
    n_violation = sum(labels)
    n_normal = len(labels) - n_violation
    print(f"  {len(corpus)} 条原始 → 过滤后 {len(filtered)} 条 | 违规: {n_violation} | 正常: {n_normal}")

    print("\n[3] 划分训练/测试集 (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  训练: {len(X_train)} | 测试: {len(X_test)}")

    print("\n[4] 训练模型...")
    model = CatoModel()
    model.train(X_train, y_train, lexicon)
    model.save()

    print("\n[5] 评估测试集...")
    evaluate(model, X_test, y_test, "测试集")

    print("\n[6] 评估训练集（检查过拟合）...")
    evaluate(model, X_train, y_train, "训练集")

    challenge_path = DATASETS_DIR / "c1.jsonl"
    if not challenge_path.exists():
        challenge_path = Path("challenges") / "c1.jsonl"
    if challenge_path.exists():
        print("\n[7] Challenge 预测...")
        run_challenge(model, challenge_path)

    print("\n[8] 交互模式（输入文本打分，回车退出）:")
    while True:
        try:
            text = input("\n请输入文本：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        result = model.score_text(text)
        icon = {"违规": "🚫", "疑似": "⚠️", "正常": "✅"}.get(result["verdict"], "")
        print(f"  {icon} [{result['verdict']}] 得分: {result['score']:.4f}")
        if result["notable_words"]:
            print("  命中词表:")
            for w in result["notable_words"]:
                print(f"    └ {w['word']} [{w['category']}]")


if __name__ == "__main__":
    main()
