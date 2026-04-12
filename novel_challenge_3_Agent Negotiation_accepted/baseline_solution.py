import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

#--- Paths ---
PUBLIC = "./dataset/public"
WORKING = "./working"
os.makedirs(WORKING, exist_ok=True)

train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))

#Feature engineering from transcript ---
#Based on problem description: protocol tokens, concession trajectory, stall patterns, sector

def extract_features(df):
    feats = pd.DataFrame(index=df.index)

    transcript = df["transcript"].fillna("")

    #Turn-level split
    turns_list = transcript.str.split(r"\s\|\|\|\s")

    #Basic counts
    feats["num_turns"] = df["num_turns"]
    feats["transcript_len"] = transcript.str.len()
    feats["word_count"] = transcript.str.split().str.len()

    #Protocol token counts (from problem description: NEXO:OFFER, COUNTER, REJECT, HOLD, etc.)
    for tok in ["NEXO:OFFER", "NEXO:BID", "NEXO:PROPOSE",
                "NEXO:COUNTER", "NEXO:REVISE", "NEXO:ADJUST",
                "NEXO:ACCEPT", "NEXO:AGREE", "NEXO:CONFIRM",
                "NEXO:REJECT", "NEXO:DECLINE", "NEXO:REFUSE",
                "NEXO:HOLD", "NEXO:PAUSE", "NEXO:DEFER",
                "NEXO:PING", "NEXO:STATUS", "NEXO:TIMEOUT", "NEXO:EXPIRE"]:
        col = "tok_" + tok.replace(":", "_").lower()
        feats[col] = transcript.str.count(re.escape(tok))

    #Grouped token families
    feats["tok_offer_family"] = feats["tok_nexo_offer"] + feats["tok_nexo_bid"] + feats["tok_nexo_propose"]
    feats["tok_counter_family"] = feats["tok_nexo_counter"] + feats["tok_nexo_revise"] + feats["tok_nexo_adjust"]
    feats["tok_accept_family"] = feats["tok_nexo_accept"] + feats["tok_nexo_agree"] + feats["tok_nexo_confirm"]
    feats["tok_reject_family"] = feats["tok_nexo_reject"] + feats["tok_nexo_decline"] + feats["tok_nexo_refuse"]
    feats["tok_stall_family"] = feats["tok_nexo_hold"] + feats["tok_nexo_pause"] + feats["tok_nexo_defer"]
    feats["tok_meta_family"] = feats["tok_nexo_ping"] + feats["tok_nexo_status"] + feats["tok_nexo_timeout"] + feats["tok_nexo_expire"]

    #Ratios (safe division)
    total_action = feats["tok_counter_family"] + feats["tok_reject_family"] + feats["tok_stall_family"] + 1
    feats["counter_ratio"] = feats["tok_counter_family"] / total_action
    feats["reject_ratio"] = feats["tok_reject_family"] / total_action
    feats["stall_ratio"] = feats["tok_stall_family"] / total_action

    #Speaker counts
    feats["buyer_turns"] = transcript.str.count(re.escape("[BUYER]"))
    feats["seller_turns"] = transcript.str.count(re.escape("[SELLER]"))
    feats["system_turns"] = transcript.str.count(re.escape("[SYSTEM]"))

    #Price extraction: find all numbers that look like prices (3+ digits)
    def extract_prices(text):
        nums = re.findall(r'\b(\d{3,5})\b', text)
        return [int(n) for n in nums]

    prices = transcript.apply(extract_prices)
    feats["price_count"] = prices.apply(len)
    feats["price_min"] = prices.apply(lambda p: min(p) if p else 0)
    feats["price_max"] = prices.apply(lambda p: max(p) if p else 0)
    feats["price_range"] = feats["price_max"] - feats["price_min"]
    feats["price_mean"] = prices.apply(lambda p: np.mean(p) if p else 0)

    #Price convergence: difference between last and first price (if multiple)
    def price_convergence(p):
        if len(p) < 2:
            return 0
        return p[-1] - p[0]
    feats["price_convergence"] = prices.apply(price_convergence)

    #Concession phrases
    concession_phrases = ["willing to compromise", "find middle ground",
                          "adjusting terms", "flexibility on our end"]
    feats["concession_count"] = sum(
        transcript.str.lower().str.count(re.escape(ph)) for ph in concession_phrases
    )

    #Stall phrases
    stall_phrases = ["check with management", "review internally",
                     "awaiting approval", "requires further discussion"]
    feats["stall_phrase_count"] = sum(
        transcript.str.lower().str.count(re.escape(ph)) for ph in stall_phrases
    )

    #Last token in transcript (the second-to-last turn of the session)
    def get_last_nexo_token(text):
        tokens = re.findall(r'NEXO:\w+', text)
        return tokens[-1] if tokens else "NONE"
    feats["last_nexo_token"] = transcript.apply(get_last_nexo_token)

    #Sector (one-hot via label encoding)
    feats["sector"] = df["sector"]

    return feats


train_feats = extract_features(train)
test_feats = extract_features(test)

#Encode categoricals
le_sector = LabelEncoder()
train_feats["sector"] = le_sector.fit_transform(train_feats["sector"])
test_feats["sector"] = le_sector.transform(test_feats["sector"])

le_last_tok = LabelEncoder()
all_last_tok = pd.concat([train_feats["last_nexo_token"], test_feats["last_nexo_token"]])
le_last_tok.fit(all_last_tok)
train_feats["last_nexo_token"] = le_last_tok.transform(train_feats["last_nexo_token"])
test_feats["last_nexo_token"] = le_last_tok.transform(test_feats["last_nexo_token"])

#Target encoding
le_label = LabelEncoder()
y_train = le_label.fit_transform(train["label"])
classes = le_label.classes_

#Feature matrix
feature_cols = [c for c in train_feats.columns]
X_train = train_feats[feature_cols].values.astype(np.float32)
X_test = test_feats[feature_cols].values.astype(np.float32)

#--- Model: LightGBM with stratified CV for robust training ---
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test), len(classes)), dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    model = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__("lightgbm").early_stopping(stopping_rounds=50, verbose=False),
            __import__("lightgbm").log_evaluation(period=0),
        ],
    )
    test_preds += model.predict_proba(X_test) / n_splits

#Final predictions
pred_labels = le_label.inverse_transform(np.argmax(test_preds, axis=1))

submission = pd.DataFrame({"id": test["id"], "label": pred_labels})
submission.to_csv(os.path.join(WORKING, "submission.csv"), index=False)

print(f"Submission shape: {submission.shape}")
print(f"Label distribution:\n{submission['label'].value_counts()}")
print("Done.")
