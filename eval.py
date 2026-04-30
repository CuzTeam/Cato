import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from main import CatoModel, load_lexicon, load_corpus, TARGET_SUBJECTS

DATASETS_DIR = Path("datasets")

ALL_SUBJECTS = list(TARGET_SUBJECTS) + ["不违规"]

LENGTH_BUCKETS = {
    "短(<10)": lambda n: n < 10,
    "中(10-50)": lambda n: 10 <= n <= 50,
    "长(>50)": lambda n: n > 50,
}


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return None
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "n": len(y_true),
        "n_pos": sum(y_true),
        "n_neg": len(y_true) - sum(y_true),
    }


def print_metrics_table(title, rows):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    print(f"  {'分组':<14} {'N':>5} {'正样本':>6} {'负样本':>6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Acc':>7}")
    print(f"  {'-'*72}")
    for name, m in rows:
        if m is None:
            print(f"  {name:<14} {'(无数据)':>5}")
        else:
            print(f"  {name:<14} {m['n']:>5} {m['n_pos']:>6} {m['n_neg']:>6} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['accuracy']:>7.4f}")


def find_lexicon_hits(text, lexicon):
    hits = []
    for w, cat in lexicon.items():
        if w in text:
            hits.append((w, cat))
    hits.sort(key=lambda x: len(x[0]), reverse=True)
    return hits[:10]


def confidence_distribution(probs, bins=None):
    if bins is None:
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    counts, _ = np.histogram(probs, bins=bins)
    return list(zip(bins[:-1], bins[1:], counts))


def print_confidence_dist(title, probs):
    dist = confidence_distribution(probs)
    print(f"\n  {title} 置信度分布 (共 {len(probs)} 条):")
    for lo, hi, cnt in dist:
        bar = "█" * cnt
        print(f"    [{lo:.1f}, {hi:.1f}) {cnt:>4}  {bar}")


def main():
    print("=" * 80)
    print(" Cato 诊断评估脚本")
    print("=" * 80)

    print("\n[1] 加载词表...")
    lexicon = load_lexicon()

    print("\n[2] 加载语料...")
    corpus = load_corpus(DATASETS_DIR / "chinese_safe.jsonl")

    records = []
    for r in corpus:
        text = r["text"]
        label_str = r["label"]
        subject = r.get("subject", "")
        if label_str == "不违规":
            records.append({"text": text, "label": 0, "subject": "不违规"})
        else:
            if subject in TARGET_SUBJECTS:
                records.append({"text": text, "label": 1, "subject": subject})

    n_violation = sum(r["label"] for r in records)
    n_normal = len(records) - n_violation
    print(f"  {len(corpus)} 条原始 → 过滤后 {len(records)} 条 | 违规: {n_violation} | 正常: {n_normal}")

    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    subjects = [r["subject"] for r in records]

    print("\n[3] 划分训练/测试集 (80/20)...")
    indices = list(range(len(texts)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )

    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    test_subjects = [subjects[i] for i in test_idx]

    print(f"  训练: {len(X_train)} | 测试: {len(X_test)}")

    print("\n[4] 训练模型...")
    model = CatoModel()
    model.train(X_train, y_train, lexicon)

    print("\n[5] 预测测试集...")
    preds, probs = model.predict(X_test)

    y_test_arr = np.array(y_test)
    preds_arr = np.array(preds)
    probs_arr = np.array(probs)

    # ── 全局指标 ──
    global_m = compute_metrics(y_test_arr, preds_arr)
    print_metrics_table("全局指标", [("全局", global_m)])

    # ── 按 subject 拆分 ──
    subject_rows = []
    for subj in ALL_SUBJECTS:
        mask = np.array([s == subj for s in test_subjects])
        if mask.sum() == 0:
            subject_rows.append((subj, None))
            continue
        if subj == "不违规":
            y_sub = np.zeros(mask.sum(), dtype=int)
            p_sub = preds_arr[mask]
        else:
            y_sub = y_test_arr[mask]
            p_sub = preds_arr[mask]
        m = compute_metrics(y_sub.tolist(), p_sub.tolist())
        subject_rows.append((subj, m))
    print_metrics_table("按 Subject 拆分指标", subject_rows)

    # ── 按文本长度拆分 ──
    text_lens = [len(t) for t in X_test]
    length_rows = []
    for bucket_name, pred_fn in LENGTH_BUCKETS.items():
        mask = np.array([pred_fn(l) for l in text_lens])
        if mask.sum() == 0:
            length_rows.append((bucket_name, None))
            continue
        m = compute_metrics(y_test_arr[mask].tolist(), preds_arr[mask].tolist())
        length_rows.append((bucket_name, m))
    print_metrics_table("按文本长度拆分指标", length_rows)

    # ── FP/FN 详细分析 ──
    fp_mask = (y_test_arr == 0) & (preds_arr == 1)
    fn_mask = (y_test_arr == 1) & (preds_arr == 0)

    fp_texts = [X_test[i] for i in range(len(X_test)) if fp_mask[i]]
    fp_probs = probs_arr[fp_mask]
    fp_subjects = [test_subjects[i] for i in range(len(X_test)) if fp_mask[i]]

    fn_texts = [X_test[i] for i in range(len(X_test)) if fn_mask[i]]
    fn_probs = probs_arr[fn_mask]
    fn_subjects = [test_subjects[i] for i in range(len(X_test)) if fn_mask[i]]

    print(f"\n{'='*80}")
    print(f" FP 详细分析 ({len(fp_texts)} 条)")
    print(f"{'='*80}")

    if len(fp_probs) > 0:
        print_confidence_dist("FP", fp_probs)

    fp_subject_counts = defaultdict(int)
    fp_hit_categories = defaultdict(int)
    print(f"\n  FP 样本明细:")
    for text, prob, subj in zip(fp_texts, fp_probs, fp_subjects):
        hits = find_lexicon_hits(text, model.lexicon)
        hit_words = [f"{w}({cat})" for w, cat in hits]
        fp_subject_counts[subj] += 1
        for _, cat in hits:
            fp_hit_categories[cat] += 1
        print(f"    p={prob:.4f} subject=[{subj}] hit={hit_words[:5]} | {text[:80]}")

    if fp_subject_counts:
        print(f"\n  FP Subject 分布:")
        for s, c in sorted(fp_subject_counts.items(), key=lambda x: -x[1]):
            print(f"    {s}: {c}")

    if fp_hit_categories:
        print(f"\n  FP 命中词表类别分布:")
        for cat, c in sorted(fp_hit_categories.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {c}")

    print(f"\n{'='*80}")
    print(f" FN 详细分析 ({len(fn_texts)} 条)")
    print(f"{'='*80}")

    if len(fn_probs) > 0:
        print_confidence_dist("FN", fn_probs)

    fn_subject_counts = defaultdict(int)
    fn_hit_categories = defaultdict(int)
    fn_no_hit = 0
    print(f"\n  FN 样本明细:")
    for text, prob, subj in zip(fn_texts, fn_probs, fn_subjects):
        hits = find_lexicon_hits(text, model.lexicon)
        hit_words = [f"{w}({cat})" for w, cat in hits]
        fn_subject_counts[subj] += 1
        for _, cat in hits:
            fn_hit_categories[cat] += 1
        if not hits:
            fn_no_hit += 1
        print(f"    p={prob:.4f} subject=[{subj}] hit={hit_words[:5]} | {text[:80]}")

    if fn_subject_counts:
        print(f"\n  FN Subject 分布:")
        for s, c in sorted(fn_subject_counts.items(), key=lambda x: -x[1]):
            print(f"    {s}: {c}")

    if fn_hit_categories:
        print(f"\n  FN 命中词表类别分布:")
        for cat, c in sorted(fn_hit_categories.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {c}")

    print(f"\n  FN 无词表命中: {fn_no_hit}/{len(fn_texts)} ({fn_no_hit/max(len(fn_texts),1)*100:.1f}%)")

    # ── 汇总 ──
    print(f"\n{'='*80}")
    print(f" 评估汇总")
    print(f"{'='*80}")
    print(f"  测试样本数:   {len(X_test)}")
    print(f"  全局 Prec:   {global_m['precision']:.4f}")
    print(f"  全局 Recall:  {global_m['recall']:.4f}")
    print(f"  全局 F1:      {global_m['f1']:.4f}")
    print(f"  全局 Acc:     {global_m['accuracy']:.4f}")
    print(f"  FP 数量:      {len(fp_texts)}")
    print(f"  FN 数量:      {len(fn_texts)}")
    print(f"  FN 无词表命中: {fn_no_hit}")

    baseline = {
        "global": global_m,
        "by_subject": {name: m for name, m in subject_rows if m is not None},
        "by_length": {name: m for name, m in length_rows if m is not None},
        "fp_count": len(fp_texts),
        "fn_count": len(fn_texts),
        "fn_no_lexicon_hit": fn_no_hit,
    }

    baseline_path = DATASETS_DIR / "baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n  基线数据已保存至 {baseline_path}")


if __name__ == "__main__":
    main()
