"""
Local evaluation: DistilBERT embeddings + structured features + LightGBM.
Shows per-fold CV scores using the challenge's weighted macro F1 metric.
Constraints: Kaggle Docker libs only, <30 min on A10G.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

PUBLIC = "./dataset/public"
train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))

# ---- Structured feature extraction ----

def extract_structured(df):
    feats = pd.DataFrame(index=df.index)
    transcript = df["transcript"].fillna("")

    feats["num_turns"] = df["num_turns"]
    feats["transcript_len"] = transcript.str.len()
    feats["word_count"] = transcript.str.split().str.len()

    for tok in ["NEXO:OFFER", "NEXO:BID", "NEXO:PROPOSE",
                "NEXO:COUNTER", "NEXO:REVISE", "NEXO:ADJUST",
                "NEXO:ACCEPT", "NEXO:AGREE", "NEXO:CONFIRM",
                "NEXO:REJECT", "NEXO:DECLINE", "NEXO:REFUSE",
                "NEXO:HOLD", "NEXO:PAUSE", "NEXO:DEFER",
                "NEXO:PING", "NEXO:STATUS", "NEXO:TIMEOUT", "NEXO:EXPIRE"]:
        col = "tok_" + tok.replace(":", "_").lower()
        feats[col] = transcript.str.count(re.escape(tok))

    feats["tok_offer_family"] = feats["tok_nexo_offer"] + feats["tok_nexo_bid"] + feats["tok_nexo_propose"]
    feats["tok_counter_family"] = feats["tok_nexo_counter"] + feats["tok_nexo_revise"] + feats["tok_nexo_adjust"]
    feats["tok_accept_family"] = feats["tok_nexo_accept"] + feats["tok_nexo_agree"] + feats["tok_nexo_confirm"]
    feats["tok_reject_family"] = feats["tok_nexo_reject"] + feats["tok_nexo_decline"] + feats["tok_nexo_refuse"]
    feats["tok_stall_family"] = feats["tok_nexo_hold"] + feats["tok_nexo_pause"] + feats["tok_nexo_defer"]
    feats["tok_meta_family"] = feats["tok_nexo_ping"] + feats["tok_nexo_status"] + feats["tok_nexo_timeout"] + feats["tok_nexo_expire"]

    total_action = feats["tok_counter_family"] + feats["tok_reject_family"] + feats["tok_stall_family"] + 1
    feats["counter_ratio"] = feats["tok_counter_family"] / total_action
    feats["reject_ratio"] = feats["tok_reject_family"] / total_action
    feats["stall_ratio"] = feats["tok_stall_family"] / total_action

    feats["buyer_turns"] = transcript.str.count(re.escape("[BUYER]"))
    feats["seller_turns"] = transcript.str.count(re.escape("[SELLER]"))
    feats["system_turns"] = transcript.str.count(re.escape("[SYSTEM]"))

    def extract_prices(text):
        return [int(n) for n in re.findall(r'\b(\d{3,5})\b', text)]

    prices = transcript.apply(extract_prices)
    feats["price_count"] = prices.apply(len)
    feats["price_min"] = prices.apply(lambda p: min(p) if p else 0)
    feats["price_max"] = prices.apply(lambda p: max(p) if p else 0)
    feats["price_range"] = feats["price_max"] - feats["price_min"]
    feats["price_mean"] = prices.apply(lambda p: np.mean(p) if p else 0)
    feats["price_convergence"] = prices.apply(lambda p: p[-1] - p[0] if len(p) >= 2 else 0)

    concession_phrases = ["willing to compromise", "find middle ground",
                          "adjusting terms", "flexibility on our end"]
    feats["concession_count"] = sum(
        transcript.str.lower().str.count(re.escape(ph)) for ph in concession_phrases
    )

    stall_phrases = ["check with management", "review internally",
                     "awaiting approval", "requires further discussion"]
    feats["stall_phrase_count"] = sum(
        transcript.str.lower().str.count(re.escape(ph)) for ph in stall_phrases
    )

    def get_last_nexo(text):
        tokens = re.findall(r'NEXO:\w+', text)
        return tokens[-1] if tokens else "NONE"
    feats["last_nexo_token"] = transcript.apply(get_last_nexo)

    feats["sector"] = df["sector"]
    return feats


# ---- DistilBERT embeddings ----

def get_distilbert_embeddings(texts, batch_size=64):
    """Extract [CLS] embeddings from DistilBERT. Uses GPU if available."""
    from transformers import DistilBertTokenizer, DistilBertModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.to(device)
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size].tolist()
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
        cls_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

        if (i // batch_size) % 20 == 0:
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embeddings)


# ---- Metric (same as grader) ----

def compute_weighted_macro_f1(y_true, y_pred):
    valid_labels = sorted({"deal_accepted", "deal_rejected", "counter_proposed", "timeout"})
    total = len(y_true)
    class_f1s = []
    class_weights = []
    for cls in valid_labels:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        cc = np.sum(y_true == cls)
        w = (total / cc) if cc > 0 else 0.0
        class_f1s.append(f1)
        class_weights.append(w)
    tw = sum(class_weights)
    if tw == 0:
        return 0.0
    class_weights = [w / tw for w in class_weights]
    return sum(f * w for f, w in zip(class_f1s, class_weights))


# ---- Build features ----

print("Extracting structured features...")
train_struct = extract_structured(train)

le_sector = LabelEncoder()
train_struct["sector"] = le_sector.fit_transform(train_struct["sector"])

le_last_tok = LabelEncoder()
le_last_tok.fit(train_struct["last_nexo_token"])
train_struct["last_nexo_token"] = le_last_tok.transform(train_struct["last_nexo_token"])

struct_cols = list(train_struct.columns)
X_struct = train_struct[struct_cols].values.astype(np.float32)

print("Extracting DistilBERT embeddings (this may take a few minutes)...")
X_emb = get_distilbert_embeddings(train["transcript"].fillna(""), batch_size=64)
print(f"Embedding shape: {X_emb.shape}")

X_all = np.hstack([X_struct, X_emb])

le_label = LabelEncoder()
y_all = le_label.fit_transform(train["label"])

# ---- CV ----

print("\nRunning 5-fold CV...")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    X_tr, X_val = X_all[tr_idx], X_all[val_idx]
    y_tr, y_val = y_all[tr_idx], y_all[val_idx]

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__("lightgbm").early_stopping(stopping_rounds=100, verbose=False),
            __import__("lightgbm").log_evaluation(period=0),
        ],
    )

    val_pred = le_label.inverse_transform(np.argmax(model.predict_proba(X_val), axis=1))
    val_true = le_label.inverse_transform(y_val)

    score = compute_weighted_macro_f1(val_true, val_pred)
    fold_scores.append(score)
    print(f"Fold {fold+1}/{n_splits}: weighted macro F1 = {score:.4f}")

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)
print(f"\nCV Summary: {mean_score:.4f} +/- {std_score:.4f}")
print(f"Per-fold: {[f'{s:.4f}' for s in fold_scores]}")
print(f"\nTarget: beat 0.816 (agent baseline)")
