import azure.functions as func
import logging
import os
import json
import random
import numpy as np
import pandas as pd
import joblib
import psycopg2
import shap
from datetime import datetime

app = func.FunctionApp()

# ── Load model artifacts (loaded once per cold start) ───────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = joblib.load(os.path.join(SCRIPT_DIR, "fraud_model.pkl"))
FEATURES = joblib.load(os.path.join(SCRIPT_DIR, "model_features.pkl"))
THRESHOLD = joblib.load(os.path.join(SCRIPT_DIR, "optimal_threshold.pkl"))
CATEGORY_MAPPINGS = joblib.load(os.path.join(SCRIPT_DIR, "category_mappings.pkl"))
SHAP_EXPLAINER = shap.TreeExplainer(MODEL)

# ── Reference data for realistic synthetic transactions ─────────────────────
PRODUCT_CODES = ["W", "C", "R", "H", "S"]
CARD_TYPES = ["debit", "credit"]
MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "online_retail",
    "restaurant", "fuel", "entertainment", "utilities", "jewelry", "pharmacy"
]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "missing"]


def generate_transaction():
    """
    Generate one synthetic transaction.
    ~5% of generated transactions follow 'fraud-like' patterns
    (high amount, late night, new card, foreign-ish email) so the
    model has realistic signal to act on.
    """
    is_fraud_pattern = random.random() < 0.05

    if is_fraud_pattern:
        amount = round(random.uniform(300, 2000), 2)
        hour = random.choice([0, 1, 2, 3, 4, 23])
        card1_count = random.randint(1, 3)        # rarely-seen card
        card_avg_ratio = round(random.uniform(4, 15), 2)
        email_domain = random.choice(["missing", "aol.com"])
    else:
        amount = round(random.uniform(5, 250), 2)
        hour = random.randint(6, 22)
        card1_count = random.randint(20, 500)
        card_avg_ratio = round(random.uniform(0.3, 2.0), 2)
        email_domain = random.choice(EMAIL_DOMAINS[:4])

    day_of_week = random.randint(0, 6)
    transaction_id = random.randint(10_000_000, 99_999_999)

    raw = {
        "TransactionID": transaction_id,
        "TransactionAmt": amount,
        "TransactionAmt_log": np.log1p(amount),
        "Amt_to_card_avg_ratio": card_avg_ratio,
        "Transaction_hour": float(hour),
        "Transaction_day": float(day_of_week),
        "card1": random.randint(1000, 18000),
        "card2": random.randint(100, 600),
        "card3": random.choice([150.0, 185.0]),
        "card5": random.choice([102.0, 117.0, 226.0]),
        "card1_count": card1_count,
        "card2_count": random.randint(1, 500),
        "addr1": random.randint(100, 500),
        "addr2": 87.0,
        "dist1": random.choice([np.nan, random.randint(0, 200)]),
        "dist2": random.choice([np.nan, random.randint(0, 200)]),
        "ProductCD": random.choice(PRODUCT_CODES),
        "card4": random.choice(["visa", "mastercard", "american express", "discover"]),
        "card6": random.choice(CARD_TYPES),
        "P_emaildomain": email_domain,
        "R_emaildomain": random.choice(EMAIL_DOMAINS),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "_actual_fraud_pattern": bool(is_fraud_pattern),
    }

    # C1-C14, D1-D5/10/11/15, V-columns: fill with plausible random values
    # Legit transactions: mostly 0/1 for C-columns, larger D-values (older account/card)
    # Fraud-pattern: higher C-columns (more linked identities), small D-values (new account)
    for c in ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12"]:
        raw[c] = float(random.choice([0,0,0,1,1,2])) if not is_fraud_pattern else float(random.randint(3, 15))

    # C13/C14: legit transactions tend to have small positive activity counts
    raw["C13"] = float(random.randint(1, 10)) if not is_fraud_pattern else float(random.choice([0, random.randint(15, 30)]))
    raw["C14"] = float(random.randint(1, 8)) if not is_fraud_pattern else float(random.choice([0, random.randint(15, 30)]))

    for d in ["D1","D2","D3","D4","D5","D10","D11","D15"]:
        raw[d] = float(random.randint(60, 400)) if not is_fraud_pattern else float(random.randint(0, 5))

    for v in ["V1","V3","V4","V5","V6","V8","V11","V13","V17","V20","V23",
              "V36","V40","V44","V47","V52","V58","V62","V70","V76","V78","V82","V91"]:
        raw[v] = float(random.choice([0,0,0,1,1])) if not is_fraud_pattern else float(random.choice([0,1,5,10]))

    for m in ["M1","M2","M3","M5","M6","M7","M8","M9"]:
        raw[m] = random.choice(["T", "T", "F"]) if not is_fraud_pattern else random.choice(["F", "missing"])

    # M4 has different categories: M0/M1/M2/missing
    raw["M4"] = random.choice(["M0", "M0", "M1"]) if not is_fraud_pattern else random.choice(["M2", "missing"])

    return raw


def encode_for_model(raw_transaction):
    """
    Convert raw transaction dict into a DataFrame row matching
    the exact feature order/encoding used during training.
    """
    row = {}
    for feat in FEATURES:
        val = raw_transaction.get(feat, np.nan)
        row[feat] = val

    df_row = pd.DataFrame([row])

    # Encode categoricals using the EXACT mappings learned during training.
    # (Encoding a single-row DataFrame with .astype('category').cat.codes
    # always yields code 0, since only one category is present — this was
    # the root cause of normal transactions scoring as fraud.)
    categorical_features = [
        'ProductCD', 'card4', 'card6',
        'P_emaildomain', 'R_emaildomain',
        'M1','M2','M3','M4','M5','M6','M7','M8','M9'
    ]
    for col in categorical_features:
        if col in df_row.columns:
            mapping = CATEGORY_MAPPINGS.get(col, {})
            raw_val = df_row[col].iloc[0]
            if pd.isna(raw_val):
                raw_val = 'missing'
            df_row[col] = mapping.get(raw_val, mapping.get('missing', -1))

    # Fill numeric NaNs same as training
    numeric_cols = [c for c in FEATURES if c not in categorical_features]
    for col in numeric_cols:
        if col in df_row.columns:
            df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(-999)

    df_row = df_row[FEATURES]
    return df_row


def get_top_shap_reason(explainer, df_row, raw_transaction):
    """
    Compute real SHAP values for this single transaction and return
    a human-readable explanation based on the top contributing features.
    SHAP on one row is fast (~10-20ms), so this is fine for a 5-min batch of 15.
    """
    shap_values = explainer.shap_values(df_row)[0]

    feature_labels = {
        "Transaction_day": "Day-of-week pattern",
        "Transaction_hour": "Unusual transaction hour",
        "TransactionAmt_log": "Transaction amount",
        "TransactionAmt": "Transaction amount",
        "Amt_to_card_avg_ratio": "Amount vs. card's average spend",
        "card1_count": "Card usage frequency",
        "card2_count": "Card usage frequency",
        "P_emaildomain": "Purchaser email domain risk",
        "R_emaildomain": "Recipient email domain risk",
        "card6": "Card type",
        "card4": "Card network",
        "ProductCD": "Product category risk",
        "D1": "Account age signal", "D2": "Account age signal",
        "D3": "Account age signal", "D4": "Account age signal",
        "D5": "Account age signal", "D10": "Account age signal",
        "D11": "Account age signal", "D15": "Account age signal",
    }
    for c in [f"C{i}" for i in range(1,15)]:
        feature_labels[c] = "Linked-identity risk signal"
    for v in ["V1","V3","V4","V5","V6","V8","V11","V13","V17","V20","V23",
              "V36","V40","V44","V47","V52","V58","V62","V70","V76","V78","V82","V91"]:
        feature_labels[v] = "Device/behavior risk signal (Vesta)"
    for m in ["M1","M2","M3","M4","M5","M6","M7","M8","M9"]:
        feature_labels[m] = "Identity match signal"

    shap_df = pd.DataFrame({
        "feature": FEATURES,
        "shap_value": shap_values
    }).sort_values("shap_value", ascending=False)

    top_positive = shap_df[shap_df["shap_value"] > 0].head(2)

    reasons = []
    for _, row_ in top_positive.iterrows():
        label = feature_labels.get(row_["feature"], row_["feature"])
        if label not in reasons:
            reasons.append(label)

    if raw_transaction["TransactionAmt"] > 300:
        amt_str = f"High amount (${raw_transaction['TransactionAmt']:.2f})"
        if amt_str not in reasons:
            reasons.insert(0, amt_str)

    if raw_transaction["Transaction_hour"] in [0,1,2,3,4,23]:
        time_str = f"Late-night transaction ({int(raw_transaction['Transaction_hour'])}:00)"
        if time_str not in reasons:
            reasons.append(time_str)

    if not reasons:
        reasons.append("Combination of multiple risk signals")

    return "; ".join(reasons[:3])


@app.timer_trigger(schedule="0 */5 * * * *", arg_name="mytimer")
def fraud_simulator(mytimer: func.TimerRequest) -> None:

    logging.info("Fraud simulator started")

    DB_CONN = os.environ["SUPABASE_CONNECTION_STRING"]
    NUM_TRANSACTIONS = 15  # generate 15 transactions every 5 minutes

    try:
        conn = psycopg2.connect(DB_CONN)
        cursor = conn.cursor()

        flagged_count = 0
        total_prob = 0.0

        for _ in range(NUM_TRANSACTIONS):
            raw = generate_transaction()
            X_row = encode_for_model(raw)

            fraud_prob = float(MODEL.predict_proba(X_row)[:, 1][0])
            is_flagged = bool(fraud_prob >= THRESHOLD)

            total_prob += fraud_prob
            if is_flagged:
                flagged_count += 1

            # Insert into transactions table
            cursor.execute("""
                INSERT INTO transactions
                (transaction_id, transaction_amt, product_cd, card_type,
                 merchant_category, transaction_hour, fraud_probability,
                 is_flagged, actual_fraud_pattern)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                raw["TransactionID"],
                raw["TransactionAmt"],
                raw["ProductCD"],
                raw["card6"],
                raw["merchant_category"],
                raw["Transaction_hour"],
                round(fraud_prob, 4),
                is_flagged,
                raw["_actual_fraud_pattern"],
            ))

            # If flagged, insert into fraud_alerts table
            if is_flagged:
                reason = get_top_shap_reason(SHAP_EXPLAINER, X_row, raw)
                cursor.execute("""
                    INSERT INTO fraud_alerts
                    (transaction_id, transaction_amt, fraud_probability,
                     merchant_category, top_reason)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    raw["TransactionID"],
                    raw["TransactionAmt"],
                    round(fraud_prob, 4),
                    raw["merchant_category"],
                    reason,
                ))

        # Update daily summary (upsert-style: delete today's row, re-insert)
        fraud_rate = flagged_count / NUM_TRANSACTIONS
        avg_prob = total_prob / NUM_TRANSACTIONS
        estimated_savings = flagged_count * 150  # avg fraud cost prevented per flag

        cursor.execute("""
            DELETE FROM daily_summary WHERE summary_date = CURRENT_DATE
        """)

        cursor.execute("""
            INSERT INTO daily_summary
            (total_transactions, total_flagged, fraud_rate, avg_fraud_probability, estimated_savings)
            SELECT
                COALESCE((SELECT COUNT(*) FROM transactions WHERE recorded_at::date = CURRENT_DATE), 0) + 0,
                COALESCE((SELECT COUNT(*) FROM transactions WHERE recorded_at::date = CURRENT_DATE AND is_flagged = TRUE), 0) + 0,
                %s, %s, %s
        """, (fraud_rate, avg_prob, estimated_savings))

        conn.commit()
        cursor.close()
        conn.close()

        logging.info(
            f"Inserted {NUM_TRANSACTIONS} transactions, "
            f"{flagged_count} flagged as fraud (threshold={THRESHOLD})"
        )

    except Exception as e:
        logging.error(f"Error: {e}")