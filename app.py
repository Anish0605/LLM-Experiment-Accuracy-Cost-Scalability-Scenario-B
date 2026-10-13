"""Scenario B — ETL benchmark with semantic value aliases and schema-name variation.

This is a separate experimental app. Scenario A remains untouched in its original repository.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="ICITIIT 2027 · Scenario B", page_icon="🧪", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1450px;padding-top:1.2rem}
[data-testid="stAppViewContainer"]{background:#f7f8fa;color:#243247}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #dbe2ea}
.hero{background:#fff;border:1px solid #dbe2ea;border-left:5px solid #eab330;border-radius:12px;padding:1.35rem 1.6rem;margin-bottom:1rem}
.kicker{font-size:.75rem;letter-spacing:.09em;font-weight:750;color:#8a6500;text-transform:uppercase}
.hero h1{font-size:2rem;color:#172b4d;margin:.3rem 0 .5rem}
[data-testid="stMetric"]{background:#fff;border:1px solid #dbe2ea;border-top:3px solid #eab330;border-radius:9px;padding:.8rem}
div.stButton>button[kind="primary"]{background:#276749;color:#fff;border-color:#276749}
</style>
""", unsafe_allow_html=True)

TITLE = "AI-Augmented Marketing Analytics: Evaluating Large Language Models in ETL Pipelines for Accuracy, Cost, and Scalability in Small-Scale Deployments"
DATA_URL = "https://www.kaggle.com/competitions/retail-return-risk-modeling"
COMPETITION = "retail-return-risk-modeling"
DEFAULT_MODEL = "grok-4.3"
DEFAULT_INPUT_RATE = 1.25
DEFAULT_OUTPUT_RATE = 2.50

FIELDS = [
    "customer_age", "product_price", "discount_percent", "product_rating",
    "past_purchase_count", "past_return_rate", "delivery_delay_days",
    "session_length_minutes", "num_product_views", "device_type",
    "product_category", "shipping_method", "payment_method", "used_coupon",
    "returned", "order_id",
]
NUMERIC_FIELDS = {
    "customer_age", "product_price", "discount_percent", "product_rating",
    "past_purchase_count", "past_return_rate", "delivery_delay_days",
    "session_length_minutes", "num_product_views", "used_coupon", "returned",
}
INTEGER_FIELDS = {"customer_age", "past_purchase_count", "num_product_views", "used_coupon", "returned"}
STRING_FIELDS = {"device_type", "product_category", "shipping_method", "payment_method"}
DECIMAL_2_FIELDS = {"product_price", "discount_percent", "product_rating", "delivery_delay_days", "session_length_minutes"}

# Schema aliases used in Scenario B. They are generated after the ground truth is frozen.
SCHEMA_ALIASES = {
    "customer_age": ["age_years", "customer age", "buyer_age"],
    "product_price": ["item_price", "price_amount", "product price"],
    "discount_percent": ["discount_pct", "promotion_percent", "discount percent"],
    "product_rating": ["rating_score", "item_rating", "product rating"],
    "past_purchase_count": ["prior_orders", "purchase_count", "past purchase count"],
    "past_return_rate": ["historical_return_ratio", "previous_return_rate", "past return rate"],
    "delivery_delay_days": ["delivery_lag", "shipping_delay_days", "delivery delay"],
    "session_length_minutes": ["session_minutes", "visit_duration_minutes", "session length"],
    "num_product_views": ["view_count", "product_views", "number_of_views"],
    "device_type": ["device_class", "customer_device", "device type"],
    "product_category": ["item_category", "merchandise_group", "product category"],
    "shipping_method": ["delivery_service", "shipping_option", "shipping method"],
    "payment_method": ["payment_type", "checkout_method", "payment method"],
    "used_coupon": ["coupon_used", "coupon_flag", "used coupon"],
    "returned": ["return_status", "was_returned", "returned_flag"],
    "order_id": ["order_reference", "transaction_order_id", "order id"],
}

# A predeclared controlled alias set. The full mapping is available to the hybrid rule
# pipeline as a lookup table; Grok receives the target schema and canonical vocabularies,
# but not this alias-to-canonical lookup table.
VALUE_ALIASES = {
    "product_category": {
        "clothing": ["apparel", "garments", "clothing category"],
        "electronics": ["consumer technology", "electronic goods", "electronics category"],
        "home": ["household goods", "home goods", "domestic products"],
        "sports": ["athletic goods", "sports equipment", "sports category"],
        "beauty": ["personal care", "beauty products", "personal-care items"],
        "toys": ["playthings", "children's toys", "toy products"],
    },
    "device_type": {
        "mobile": ["handheld device", "mobile phone", "smartphone"],
        "desktop": ["workstation", "desktop computer", "personal computer"],
        "tablet": ["slate device", "tablet computer", "touchscreen tablet"],
    },
    "shipping_method": {
        "standard": ["standard delivery", "regular post", "standard shipping"],
        "express": ["express delivery", "priority shipping", "express shipping"],
        "same_day": ["same-day delivery", "delivery today", "same day shipping"],
    },
    "payment_method": {
        "credit_card": ["credit card payment", "credit card", "card credit"],
        "debit_card": ["debit card payment", "debit card", "card debit"],
        "paypal": ["PayPal wallet", "PayPal payment", "digital wallet PayPal"],
        "apple_pay": ["Apple Pay wallet", "Apple Pay payment", "mobile wallet Apple Pay"],
        "google_pay": ["Google Pay wallet", "Google Pay payment", "mobile wallet Google Pay"],
        "cash_on_delivery": ["cash on delivery payment", "cash on delivery", "pay on delivery"],
        "bank_transfer": ["bank transfer payment", "bank transfer", "direct bank transfer"],
    },
}

# Field meaning descriptions help the model infer header aliases consistently.
FIELD_DESCRIPTIONS = {
    "customer_age": "customer age in years",
    "product_price": "item/product price as a numeric amount",
    "discount_percent": "discount percentage from 0 to 100",
    "product_rating": "product rating from 1 to 5",
    "past_purchase_count": "number of previous purchases",
    "past_return_rate": "historical return rate as a fraction from 0 to 1",
    "delivery_delay_days": "delivery delay duration in days",
    "session_length_minutes": "session duration in minutes",
    "num_product_views": "number of product views",
    "device_type": "device class used by the customer",
    "product_category": "canonical product category",
    "shipping_method": "shipping/delivery method",
    "payment_method": "payment method",
    "used_coupon": "whether a coupon was used, encoded 0 or 1",
    "returned": "whether the order was returned, encoded 0 or 1",
    "order_id": "order identifier as a string",
}


def setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip() or default


def api_key() -> str:
    return setting("XAI_API_KEY", "")


def kaggle_ready() -> bool:
    return bool(setting("KAGGLE_USERNAME") and setting("KAGGLE_KEY"))


def download_train_csv() -> tuple[bytes, str]:
    username, key = setting("KAGGLE_USERNAME"), setting("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError("Add KAGGLE_USERNAME and KAGGLE_KEY to this Streamlit app's Secrets.")
    env = os.environ.copy()
    env["KAGGLE_USERNAME"], env["KAGGLE_KEY"] = username, key
    with tempfile.TemporaryDirectory(prefix="scenario_b_kaggle_") as tmp:
        command = ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", "train.csv", "-p", tmp]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "No details returned")[-1500:]
            raise RuntimeError("Kaggle download failed. Confirm your legacy username/key and competition access. " + detail)
        direct = Path(tmp) / "train.csv"
        zips = list(Path(tmp).glob("*.zip"))
        if direct.exists():
            data = direct.read_bytes()
        elif zips:
            with zipfile.ZipFile(zips[0]) as zf:
                match = next((n for n in zf.namelist() if Path(n).name.lower() == "train.csv"), None)
                if not match:
                    raise RuntimeError("Downloaded archive does not contain train.csv.")
                data = zf.read(match)
        else:
            raise RuntimeError("Kaggle command completed but train.csv was not found.")
        return data, "train.csv (Kaggle API)"


@st.cache_data(show_spinner=False, max_entries=2)
def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def eligibility(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = [f for f in FIELDS if f not in df.columns]
    if missing:
        raise ValueError("CSV is missing required fields: " + ", ".join(missing))
    d = df[FIELDS].copy()
    for field in NUMERIC_FIELDS:
        d[field] = pd.to_numeric(d[field], errors="coerce")
    rules = {
        "All 16 required fields present": d[FIELDS].notna().all(axis=1),
        "Customer age between 18 and 100": d["customer_age"].between(18, 100),
        "Product price greater than zero": d["product_price"].gt(0),
        "Discount between 0 and 100": d["discount_percent"].between(0, 100),
        "Product rating between 1 and 5": d["product_rating"].between(1, 5),
        "Purchase count non-negative": d["past_purchase_count"].ge(0),
        "Return rate between 0 and 1": d["past_return_rate"].between(0, 1),
        "Delivery delay non-negative": d["delivery_delay_days"].ge(0),
        "Session length non-negative": d["session_length_minutes"].ge(0),
        "Product views non-negative": d["num_product_views"].ge(0),
        "Coupon flag is binary": d["used_coupon"].isin([0, 1]),
        "Return flag is binary": d["returned"].isin([0, 1]),
    }
    for field in STRING_FIELDS:
        rules[field + " is non-empty"] = d[field].astype("string").str.strip().ne("").fillna(False)
    rules["Order ID is unique"] = ~d["order_id"].astype("string").str.strip().duplicated(keep="first")
    combined = pd.Series(True, index=d.index)
    for rule in rules.values():
        combined &= rule.fillna(False)
    return d.loc[combined].copy(), {name: int(mask.fillna(False).sum()) for name, mask in rules.items()}


def number_from(value: Any, percent_fraction: bool = False) -> float | None:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    txt = str(value).strip().lower().replace(",", "")
    has_pct = "%" in txt
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", txt)
    if not match:
        return None
    number = float(match.group(0))
    return number / 100 if (has_pct and percent_fraction) else number


def canonicalize(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    d = dict(row)
    out = {
        "customer_age": int(round(float(d["customer_age"]))),
        "product_price": round(float(d["product_price"]), 2),
        "discount_percent": round(float(d["discount_percent"]), 2),
        "product_rating": round(float(d["product_rating"]), 2),
        "past_purchase_count": int(round(float(d["past_purchase_count"]))),
        "past_return_rate": round(float(d["past_return_rate"]), 4),
        "delivery_delay_days": round(float(d["delivery_delay_days"]), 2),
        "session_length_minutes": round(float(d["session_length_minutes"]), 2),
        "num_product_views": int(round(float(d["num_product_views"]))),
        "used_coupon": int(round(float(d["used_coupon"]))),
        "returned": int(round(float(d["returned"]))),
        "order_id": str(d["order_id"]).strip(),
    }
    for field in STRING_FIELDS:
        out[field] = str(d[field]).strip().lower()
    return {field: out[field] for field in FIELDS}


def choose_alias(field: str, canonical_value: str, row_index: int) -> str:
    choices = VALUE_ALIASES.get(field, {}).get(canonical_value, [])
    if choices:
        return choices[(row_index // 2) % len(choices)]
    # Fallback for any additional canonical enum values in the source data.
    suffix = {
        "product_category": " category",
        "device_type": " device",
        "shipping_method": " delivery",
        "payment_method": " payment option",
    }.get(field, "")
    return canonical_value.replace("_", " ") + suffix


def corrupt_scenario_b(gold: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Apply A's formatting variants plus B's controlled aliases and alternate headers."""
    values = dict(gold)
    if row_index % 2 == 0:
        values["customer_age"] = f'{gold["customer_age"]} years'
        values["product_price"] = f'price {gold["product_price"]:.2f}'
        values["discount_percent"] = f'{gold["discount_percent"]:.2f} %'
        values["product_rating"] = f'{gold["product_rating"]:.2f} stars'
        values["past_purchase_count"] = f'{gold["past_purchase_count"]} purchases'
        values["past_return_rate"] = f'{gold["past_return_rate"] * 100:.2f}%'
        values["delivery_delay_days"] = f'{gold["delivery_delay_days"]:.2f} days'
        values["session_length_minutes"] = f'{gold["session_length_minutes"]:.2f} min'
        values["num_product_views"] = f'{gold["num_product_views"]} views'
    else:
        values["customer_age"] = str(gold["customer_age"])
        values["product_price"] = f'{gold["product_price"]:.2f} currency-units'
        values["discount_percent"] = f'{gold["discount_percent"]:.2f} percent'
        values["product_rating"] = f'rating: {gold["product_rating"]:.2f}/5'
        values["past_purchase_count"] = f'count={gold["past_purchase_count"]}'
        values["past_return_rate"] = f'rate={gold["past_return_rate"]:.4f}'
        values["delivery_delay_days"] = f'delay {gold["delivery_delay_days"]:.2f}'
        values["session_length_minutes"] = f'{gold["session_length_minutes"]:.2f} minutes'
        values["num_product_views"] = f'views={gold["num_product_views"]}'
    values["used_coupon"] = " YES " if gold["used_coupon"] == 1 else "no"
    values["returned"] = "Returned" if gold["returned"] == 1 else "Not returned"
    values["order_id"] = f'ORD-{gold["order_id"]}'
    for field in STRING_FIELDS:
        value = str(gold[field]).strip().lower()
        if (row_index + FIELDS.index(field)) % 4 != 0:
            values[field] = choose_alias(field, value, row_index)
        else:
            values[field] = value.upper() if row_index % 2 == 0 else value.title()

    # Keep a canonical field name for some headers; otherwise use one of the alternatives.
    record = {"record_key": f"R{row_index + 1:05d}"}
    for position, field in enumerate(FIELDS):
        if (row_index + position) % 3 == 0:
            key = field
        else:
            alternatives = SCHEMA_ALIASES[field]
            key = alternatives[(row_index + position) % len(alternatives)]
        record[key] = values[field]
    return record


def alias_lookup() -> dict[str, dict[str, str]]:
    reverse: dict[str, dict[str, str]] = {}
    for field in STRING_FIELDS:
        reverse[field] = {}
        for canonical, aliases in VALUE_ALIASES.get(field, {}).items():
            for alias in aliases:
                reverse[field][alias.strip().lower()] = canonical
    return reverse


def reverse_schema_aliases(row: dict[str, Any]) -> dict[str, Any]:
    reverse = {alias.strip().lower(): field for field, aliases in SCHEMA_ALIASES.items() for alias in aliases}
    canonical = {"record_key": row.get("record_key", "")}
    for key, value in row.items():
        if key == "record_key":
            continue
        normalized_key = str(key).strip().lower()
        field = normalized.get(normalized_key, normalized_key) if False else normalized_key
        field = reverse.get(field, field.replace(" ", "_"))
        if field in FIELDS:
            canonical[field] = value
    return canonical


def parse_numeric(value: Any, field: str) -> Any:
    n = number_from(value, percent_fraction=(field == "past_return_rate"))
    if n is None:
        return value
    if field in INTEGER_FIELDS:
        return int(round(n))
    if field in DECIMAL_2_FIELDS:
        return round(n, 2)
    if field == "past_return_rate":
        return round(n, 4)
    return n


def rule_transform(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        value = row.get(field)
        if field == "used_coupon":
            low = str(value).strip().lower()
            out[field] = 1 if low in {"yes", "true", "1"} else 0 if low in {"no", "false", "0"} else value
        elif field == "returned":
            low = str(value).strip().lower()
            out[field] = 1 if low in {"returned", "yes", "true", "1"} else 0 if low in {"not returned", "no", "false", "0"} else value
        else:
            out[field] = parse_numeric(value, field)
    for field in STRING_FIELDS:
        out[field] = str(row.get(field, "")).strip().lower()
    out["order_id"] = re.sub(r"^ord-", "", str(row.get("order_id", "")).strip(), flags=re.IGNORECASE)
    return {field: out.get(field) for field in FIELDS}


def alias_aware_transform(row: dict[str, Any]) -> dict[str, Any]:
    canonical = reverse_schema_aliases(row)
    reverse_values = alias_lookup()
    for field in STRING_FIELDS:
        value = str(canonical.get(field, "")).strip().lower()
        canonical_value = reverse_values.get(field, {}).get(value)
        if canonical_value:
            canonical[field] = canonical_value
        else:
            # Normalize fallback aliases generated for values outside the declared alias set.
            value_without_suffix = re.sub(r"( category| device| delivery| payment option)$", "", value)
            canonical[field] = value_without_suffix.replace(" ", "_") if value_without_suffix else value
    return rule_transform(canonical)


def equal_value(field: str, predicted: Any, target: Any) -> bool:
    if predicted is None:
        return False
    if field in NUMERIC_FIELDS:
        p = number_from(predicted, percent_fraction=(field == "past_return_rate"))
        t = number_from(target, percent_fraction=(field == "past_return_rate"))
        if p is None or t is None:
            return False
        tolerance = 1e-6 if field in INTEGER_FIELDS or field == "past_return_rate" else 0.001
        return abs(p - t) <= tolerance
    p, t = str(predicted).strip().lower(), str(target).strip().lower()
    if field == "order_id":
        p = re.sub(r"^ord-", "", p, flags=re.IGNORECASE)
        t = re.sub(r"^ord-", "", t, flags=re.IGNORECASE)
    return p == t


def evaluate(gold: dict[str, dict[str, Any]], outputs: dict[str, dict[str, Any]], keys: list[str], method: str) -> dict[str, Any]:
    correct_fields, correct_records = 0, 0
    for key in keys:
        predicted = outputs.get(key, {}) if isinstance(outputs.get(key, {}), dict) else {}
        checks = [equal_value(field, predicted.get(field), gold[key].get(field)) for field in FIELDS]
        correct_fields += sum(checks)
        correct_records += int(bool(checks) and all(checks))
    total = len(keys) * len(FIELDS)
    return {
        "sample_size": len(keys), "method": method,
        "field_accuracy_pct": round(100 * correct_fields / total, 2) if total else 0,
        "record_accuracy_pct": round(100 * correct_records / len(keys), 2) if keys else 0,
        "correct_fields": correct_fields, "total_fields": total,
        "fully_correct_records": correct_records,
    }


def llm_prompt(rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> str:
    enum_fields = ["device_type", "product_category", "shipping_method", "payment_method"]
    vocab = {f: sorted({str(gold[k][f]) for k in gold}) for f in enum_fields}
    descriptions = {f: FIELD_DESCRIPTIONS[f] for f in FIELDS}
    return (
        "Scenario B controlled ETL benchmark. Normalize records whose column names may be aliases "
        "and whose categorical values may be semantically equivalent synonyms. Map them to the exact "
        "canonical schema and values. Use semantic meaning from header and value context. Do not guess "
        "missing facts, predict returns, or invent new categories. Preserve record_key exactly.\n\n"
        "Canonical field meanings: " + json.dumps(descriptions, ensure_ascii=False) + "\n"
        "Allowed canonical values for category-like fields (a guide, not an alias lookup): " +
        json.dumps(vocab, ensure_ascii=False) + "\n"
        "Numeric conventions: customer_age integer; product_price 2 decimals; discount_percent percentage "
        "0-100; product_rating 1-5 with 2 decimals; past_purchase_count integer; past_return_rate fraction "
        "0-1 with 4 decimals (a percent sign means convert percent to fraction); delivery_delay_days and "
        "session_length_minutes 2 decimals; num_product_views integer; used_coupon and returned are integers "
        "0/1; order_id is a string with an optional ORD- prefix removed. Normalize categorical outputs to the "
        "exact allowed canonical spelling (lowercase; underscores where present).\n\n"
        "Return valid JSON only in the shape {\"records\":[{\"record_key\":\"...\", plus all 16 canonical fields}]}. "
        "All rows must include every canonical field.\nINPUT:\n" +
        json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    )


def call_grok(client: OpenAI, rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]], model: str, in_rate: float, out_rate: float):
    started = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Normalize records into an exact schema. Return valid JSON only."},
            {"role": "user", "content": llm_prompt(rows, gold)},
        ],
        temperature=0,
        max_tokens=3500,
        response_format={"type": "json_object"},
    )
    elapsed = time.perf_counter() - started
    raw = completion.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    result_rows = parsed.get("records", []) if isinstance(parsed, dict) else parsed
    outputs = {}
    for result in result_rows if isinstance(result_rows, list) else []:
        if isinstance(result, dict) and result.get("record_key") is not None:
            k = str(result["record_key"])
            outputs[k] = {f: result.get(f) for f in FIELDS}
    usage = getattr(completion, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    has_usage = usage is not None and getattr(usage, "prompt_tokens", None) is not None and getattr(usage, "completion_tokens", None) is not None
    cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000 if has_usage else None
    meta = {
        "call_id": "call_" + uuid.uuid4().hex[:12],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model, "record_keys": [str(r.get("record_key", "")) for r in rows],
        "elapsed_sec": round(elapsed, 5), "input_tokens": input_tokens,
        "output_tokens": output_tokens, "has_usage": has_usage,
        "estimated_service_cost_usd": round(cost, 8) if cost is not None else None,
        "batch_records": len(rows), "error": None, "raw_response": raw,
    }
    return outputs, meta


def failed_call(rows: list[dict[str, Any]], model: str, exc: Exception, elapsed: float):
    return {
        "call_id": "call_" + uuid.uuid4().hex[:12],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model, "record_keys": [str(r.get("record_key", "")) for r in rows],
        "elapsed_sec": round(elapsed, 5), "input_tokens": 0, "output_tokens": 0,
        "has_usage": False, "estimated_service_cost_usd": None, "batch_records": len(rows),
        "error": type(exc).__name__ + ": " + str(exc)[:400], "raw_response": None,
    }


def initialize():
    defaults = {
        "source_bytes": None, "source_name": None, "source_hash": None,
        "source_rows": 0, "eligible_rows": 0, "eligibility_counts": {},
        "gold": {}, "corrupted": {}, "frozen_outputs": {}, "frozen_times": {},
        "hybrid_outputs": {}, "hybrid_times": {}, "llm_outputs": {}, "llm_calls": [],
        "model_meta": {}, "seed": 42, "built_at": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_benchmark():
    for key, val in {
        "gold": {}, "corrupted": {}, "frozen_outputs": {}, "frozen_times": {},
        "hybrid_outputs": {}, "hybrid_times": {}, "llm_outputs": {}, "llm_calls": [],
        "model_meta": {}, "built_at": None,
    }.items():
        st.session_state[key] = val


def main():
    initialize()
    model = setting("XAI_MODEL", DEFAULT_MODEL)
    key = api_key()
    st.markdown(
        "<div class='hero'><div class='kicker'>ICITIIT 2027 · Scenario B</div>"
        "<h1>ETL Benchmark: Unseen Aliases & Schema Variations</h1>"
        "<div>Compare frozen rules, alias-aware deterministic rules, and Grok-assisted ETL on the same controlled records.</div>"
        "<p><b>Research title:</b> " + TITLE + "</p></div>", unsafe_allow_html=True
    )
    with st.sidebar:
        st.markdown("### Model & API")
        st.caption("Model ID: " + model)
        if key:
            st.success("XAI_API_KEY detected")
        else:
            st.error("XAI_API_KEY not configured")
        if st.button("Test Grok connection", disabled=not key, use_container_width=True):
            try:
                available = [m.id for m in OpenAI(api_key=key, base_url="https://api.x.ai/v1").models.list().data]
                if model in available:
                    st.success("Connection OK; model " + model + " is listed.")
                else:
                    st.warning("API connected, but model ID was not listed. Verify access and spelling.")
            except Exception as exc:
                st.error(type(exc).__name__ + ": " + str(exc)[:250])
        in_rate = st.number_input("Input USD / 1M tokens", min_value=0.0, value=DEFAULT_INPUT_RATE, step=0.025, format="%.4f")
        out_rate = st.number_input("Output USD / 1M tokens", min_value=0.0, value=DEFAULT_OUTPUT_RATE, step=0.05, format="%.4f")
        st.caption("Reference-priced token cost is an estimate, not actual billing or total ownership cost.")
        st.markdown("---")
        for label, done in [
            ("Dataset loaded", bool(st.session_state.source_bytes)),
            ("Scenario B benchmark built", bool(st.session_state.gold)),
            ("Frozen baseline run", bool(st.session_state.frozen_outputs)),
            ("Alias-aware rules run", bool(st.session_state.hybrid_outputs)),
            ("Grok outputs collected", bool(st.session_state.llm_outputs)),
        ]:
            st.write(("✅ " if done else "⬜ ") + label)

    colsrc, colupload = st.columns([1, 1])
    with colsrc:
        st.markdown("#### Dataset source")
        st.markdown("The app downloads train.csv through Kaggle API. Accept competition rules first: " + DATA_URL)
        if kaggle_ready():
            st.success("Kaggle username/key detected in Streamlit Secrets.")
        else:
            st.warning("Add KAGGLE_USERNAME and KAGGLE_KEY under Streamlit Secrets.")
        if st.button("Download train.csv from Kaggle", type="primary", disabled=not kaggle_ready()):
            try:
                with st.spinner("Downloading train.csv from Kaggle…"):
                    data, name = download_train_csv()
                    st.session_state.source_bytes = data
                    st.session_state.source_name = name
                    st.session_state.source_hash = hashlib.sha256(data).hexdigest()
                st.success("Downloaded " + name + " (" + format(len(data), ",") + " bytes).")
            except Exception as exc:
                st.error("Kaggle download failed: " + str(exc))
    with colupload:
        uploaded = st.file_uploader("Optional fallback: upload train.csv", type=["csv"])
        if uploaded is not None:
            data = uploaded.getvalue()
            st.session_state.source_bytes = data
            st.session_state.source_name = uploaded.name
            st.session_state.source_hash = hashlib.sha256(data).hexdigest()
            st.caption("Manual upload selected; this overrides the downloaded file.")

    source = None
    eligible = None
    if st.session_state.source_bytes:
        try:
            source = read_csv(st.session_state.source_bytes)
            eligible, counts = eligibility(source)
            st.session_state.source_rows = len(source)
            st.session_state.eligible_rows = len(eligible)
            st.session_state.eligibility_counts = counts
        except Exception as exc:
            st.error("Dataset validation failed: " + str(exc))
            source = None
            eligible = None

    tabs = st.tabs(["Overview", "Dataset & benchmark", "Run pipelines", "Results", "Audit & export", "Protocol"])
    t0, t1, t2, t3, t4, t5 = tabs
    with t0:
        st.subheader("Scenario B at a glance")
        a, b, c, d = st.columns(4)
        a.metric("Source rows", f"{len(source):,}" if source is not None else "—")
        b.metric("Eligible rows", f"{len(eligible):,}" if eligible is not None else "—")
        c.metric("Benchmark size", len(st.session_state.gold) or "Not built")
        d.metric("Grok records processed", len(st.session_state.llm_outputs))
        st.markdown("#### Research objective")
        st.write("Test whether LLM assistance adds value when records contain semantic category aliases and alternate field names not handled by the original Scenario A rules.")
        st.info("Scenario B is a controlled challenge. The alias vocabulary and ground truth are specified by the benchmark generator. The results do not automatically generalize to all real production data.")
        st.markdown("#### Experimental arms")
        st.write("1. Frozen rules: the Scenario A transformation logic, without new header or semantic alias maps.")
        st.write("2. Alias-aware rules: the same deterministic parser plus explicit, predeclared schema/value lookup dictionaries.")
        st.write("3. Grok-assisted: the LLM is told the canonical schema, field meanings, and canonical vocabulary, but not the alias lookup table.")

    with t1:
        st.subheader("1 · Validate data and build Scenario B sample")
        if eligible is None:
            st.info("Download or upload train.csv first.")
        else:
            a, b, c = st.columns(3)
            a.metric("Source rows", f"{len(source):,}")
            b.metric("Eligible rows", f"{len(eligible):,}")
            c.metric("Excluded rows", f"{len(source) - len(eligible):,}")
            with st.expander("Eligibility rule counts"):
                st.dataframe(pd.DataFrame({"Rule": list(st.session_state.eligibility_counts), "Rows passing": list(st.session_state.eligibility_counts.values())}), use_container_width=True, hide_index=True)
            left, right = st.columns([1, 1.3])
            with left:
                size = st.selectbox("Records in benchmark", [20, 50, 100, 200], index=2)
                seed = st.number_input("Random seed", 1, 999999, 42, 1)
                st.caption("Use the same seed for reproducibility. Building a new benchmark clears current outputs.")
                if st.button("Build / reset Scenario B benchmark", type="primary", disabled=len(eligible) < size, use_container_width=True):
                    sample = eligible.sample(n=int(size), random_state=int(seed)).reset_index(drop=True)
                    gold, corrupted = {}, {}
                    for i, row in sample.iterrows():
                        rid = f"R{i+1:05d}"
                        canonical = canonicalize(row)
                        gold[rid] = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in canonical.items()}
                        corrupted[rid] = corrupt_scenario_b(canonical, i)
                    clear_benchmark()
                    st.session_state.gold = gold
                    st.session_state.corrupted = corrupted
                    st.session_state.seed = int(seed)
                    st.session_state.built_at = datetime.now(timezone.utc).isoformat()
                    st.success(f"Created Scenario B benchmark with {size} records (seed {seed}).")
                    st.rerun()
            with right:
                st.markdown("#### Source preview")
                st.dataframe(source.head(6), use_container_width=True, hide_index=True)
                st.caption("Source SHA-256: " + str(st.session_state.source_hash))
            if st.session_state.gold:
                st.success(f"Scenario B benchmark ready: {len(st.session_state.gold)} records; seed {st.session_state.seed}.")
                with st.expander("Inspect one ground-truth record and its Scenario B input"):
                    rid = st.selectbox("Record", list(st.session_state.gold), key="inspect_b_record")
                    x, y = st.columns(2)
                    x.markdown("**Ground truth (canonical)**")
                    x.json(st.session_state.gold[rid])
                    y.markdown("**Scenario B input (altered headers and values)**")
                    y.json(st.session_state.corrupted[rid])

    with t2:
        st.subheader("2 · Run all three pipelines")
        if not st.session_state.gold:
            st.info("Build the benchmark first.")
        else:
            ids = list(st.session_state.gold)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### A. Frozen rules")
                st.write("Same core rule parser as Scenario A; no new alias/header mapping.")
                if st.button("Run frozen rule baseline", use_container_width=True):
                    outputs, times = {}, {}
                    for rid in ids:
                        start = time.perf_counter()
                        outputs[rid] = rule_transform(st.session_state.corrupted[rid])
                        times[rid] = time.perf_counter() - start
                    st.session_state.frozen_outputs, st.session_state.frozen_times = outputs, times
                    st.success("Frozen baseline finished.")
                if st.session_state.frozen_outputs:
                    st.metric("Field accuracy", f'{evaluate(st.session_state.gold, st.session_state.frozen_outputs, ids, "Frozen rules")["field_accuracy_pct"]:.2f}%')
            with col2:
                st.markdown("#### B. Alias-aware rules")
                st.write("Deterministic parser plus predeclared schema and value mapping tables.")
                if st.button("Run alias-aware rules", use_container_width=True):
                    outputs, times = {}, {}
                    for rid in ids:
                        start = time.perf_counter()
                        outputs[rid] = alias_aware_transform(st.session_state.corrupted[rid])
                        times[rid] = time.perf_counter() - start
                    st.session_state.hybrid_outputs, st.session_state.hybrid_times = outputs, times
                    st.success("Alias-aware rule run finished.")
                if st.session_state.hybrid_outputs:
                    st.metric("Field accuracy", f'{evaluate(st.session_state.gold, st.session_state.hybrid_outputs, ids, "Alias-aware rules")["field_accuracy_pct"]:.2f}%')
            with col3:
                st.markdown("#### C. Grok-assisted")
                st.write("Model uses canonical schema semantics, not the explicit alias mapping table.")
                target = st.select_slider("LLM records to process", options=[n for n in [20, 50, 100, 200] if n <= len(ids)], value=min(20, len(ids)))
                batch_size = st.selectbox("Records per API request", [1, 5], index=1)
                st.caption("Start with 20; extend only after reviewing costs and errors.")
                if st.button("Run / extend Grok experiment", disabled=not key, type="primary", use_container_width=True):
                    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
                    targets = ids[:int(target)]
                    missing = [rid for rid in targets if rid not in st.session_state.llm_outputs]
                    if not missing:
                        st.info("Selected records already have outputs. Increase target size to extend.")
                    else:
                        batches = [missing[i:i+int(batch_size)] for i in range(0, len(missing), int(batch_size))]
                        progress = st.progress(0.0)
                        status = st.empty()
                        for bi, batch_ids in enumerate(batches, start=1):
                            rows = [st.session_state.corrupted[rid] for rid in batch_ids]
                            start = time.perf_counter()
                            try:
                                out, meta = call_grok(client, rows, st.session_state.gold, model, float(in_rate), float(out_rate))
                            except Exception as exc:
                                out = {}
                                meta = failed_call(rows, model, exc, time.perf_counter() - start)
                            st.session_state.llm_calls.append(meta)
                            for rid in batch_ids:
                                st.session_state.llm_outputs[rid] = out.get(rid, {})
                                st.session_state.model_meta[rid] = meta
                            status.write(f"Batch {bi}/{len(batches)} · {sum(bool(out.get(rid)) for rid in batch_ids)} valid record outputs returned.")
                            progress.progress(bi / len(batches))
                        st.rerun()
            st.markdown("---")
            st.write("Benchmark size: " + str(len(ids)) + " · Frozen outputs: " + str(len(st.session_state.frozen_outputs)) + " · Alias-aware outputs: " + str(len(st.session_state.hybrid_outputs)) + " · Grok outputs: " + str(len(st.session_state.llm_outputs)))

    with t3:
        st.subheader("3 · Results for completed sample sizes")
        if not st.session_state.gold:
            st.info("Build Scenario B first.")
        else:
            ids = list(st.session_state.gold)
            summary = []
            for n in [20, 50, 100, 200]:
                if n > len(ids):
                    continue
                subset = ids[:n]
                for method, outs, times in [
                    ("Frozen rules", st.session_state.frozen_outputs, st.session_state.frozen_times),
                    ("Alias-aware rules", st.session_state.hybrid_outputs, st.session_state.hybrid_times),
                ]:
                    if all(rid in outs for rid in subset):
                        ev = evaluate(st.session_state.gold, outs, subset, method)
                        seconds = sum(times.get(rid, 0) for rid in subset)
                        summary.append({**ev, "processing_seconds": seconds, "records_per_second": n / seconds if seconds > 0 else None, "input_tokens": 0, "output_tokens": 0, "estimated_service_cost_usd": 0.0, "api_errors": 0})
                if all(rid in st.session_state.llm_outputs for rid in subset):
                    ev = evaluate(st.session_state.gold, st.session_state.llm_outputs, subset, "LLM-assisted")
                    subset_set = set(subset)
                    calls = [x for x in st.session_state.llm_calls if x.get("record_keys") and set(x["record_keys"]).issubset(subset_set)]
                    seconds = sum(float(x.get("elapsed_sec") or 0) for x in calls)
                    complete_usage = bool(calls) and all(x.get("has_usage", False) for x in calls)
                    cost = sum(float(x.get("estimated_service_cost_usd") or 0) for x in calls) if complete_usage else None
                    summary.append({**ev, "processing_seconds": seconds, "records_per_second": n / seconds if seconds > 0 else None, "input_tokens": sum(int(x.get("input_tokens") or 0) for x in calls), "output_tokens": sum(int(x.get("output_tokens") or 0) for x in calls), "estimated_service_cost_usd": cost, "api_errors": sum(bool(x.get("error")) for x in calls)})
            if not summary:
                st.info("Run at least one rule pipeline and a Grok sample to populate the comparison.")
            else:
                sdf = pd.DataFrame(summary)
                max_n = int(sdf["sample_size"].max())
                maxrows = sdf[sdf["sample_size"] == max_n]
                metrics = st.columns(min(4, len(maxrows)))
                for i, (_, row) in enumerate(maxrows.iterrows()):
                    metrics[i].metric(str(row["method"]) + " accuracy", f'{row["field_accuracy_pct"]:.2f}%')
                st.dataframe(sdf[["sample_size", "method", "field_accuracy_pct", "record_accuracy_pct", "processing_seconds", "records_per_second", "input_tokens", "output_tokens", "estimated_service_cost_usd", "api_errors"]].round(6), use_container_width=True, hide_index=True)
                acc_long = sdf.melt(id_vars=["sample_size", "method"], value_vars=["field_accuracy_pct", "record_accuracy_pct"], var_name="measure", value_name="accuracy_pct")
                fig = px.bar(acc_long, x="measure", y="accuracy_pct", color="method", barmode="group", facet_col="sample_size", range_y=[0, 100], title="Accuracy by method and completed sample size")
                st.plotly_chart(fig, use_container_width=True)
                fig = px.line(sdf, x="sample_size", y="processing_seconds", color="method", markers=True, title="Observed processing wall time", labels={"sample_size": "Records", "processing_seconds": "Seconds"})
                st.plotly_chart(fig, use_container_width=True)
                llm_rows = sdf[sdf["method"] == "LLM-assisted"]
                if not llm_rows.empty and llm_rows["estimated_service_cost_usd"].notna().any():
                    fig = px.bar(llm_rows, x="sample_size", y="estimated_service_cost_usd", title="Grok service-equivalent token cost", labels={"sample_size": "Records", "estimated_service_cost_usd": "USD"})
                    st.plotly_chart(fig, use_container_width=True)
                st.caption("Frozen rules deliberately do not contain Scenario B alias tables. Alias-aware rules include the mapping dictionaries; Grok receives canonical field meanings and vocabulary but not the alias lookup table. Report these experimental arms separately.")
                st.markdown("#### Field-level comparison on largest completed sample")
                field_rows = []
                largest_ids = ids[:max_n]
                method_outputs = [
                    ("Frozen rules", st.session_state.frozen_outputs),
                    ("Alias-aware rules", st.session_state.hybrid_outputs),
                    ("LLM-assisted", st.session_state.llm_outputs),
                ]
                for field in FIELDS:
                    for method, outs in method_outputs:
                        if all(rid in outs for rid in largest_ids):
                            good = sum(equal_value(field, outs[rid].get(field), st.session_state.gold[rid][field]) for rid in largest_ids)
                            field_rows.append({"field": field, "method": method, "accuracy_pct": 100 * good / len(largest_ids)})
                if field_rows:
                    fdf = pd.DataFrame(field_rows)
                    fig = px.bar(fdf, x="accuracy_pct", y="field", color="method", barmode="group", orientation="h", range_x=[0, 100], title="Field-level accuracy on largest completed sample")
                    st.plotly_chart(fig, use_container_width=True)
                with st.expander("Inspect Grok API errors"):
                    errors = [x for x in st.session_state.llm_calls if x.get("error")]
                    if errors:
                        st.dataframe(pd.DataFrame(errors)[["call_id", "model_id", "record_keys", "elapsed_sec", "error"]], use_container_width=True, hide_index=True)
                    else:
                        st.success("No API errors logged.")

    with t4:
        st.subheader("4 · Export Scenario B evidence")
        if not st.session_state.gold:
            st.info("Build the benchmark first.")
        else:
            payload = {
                "metadata": {
                    "scenario": "B: schema-name variation and semantic value aliases",
                    "research_title": TITLE, "model_id": model, "seed": st.session_state.seed,
                    "benchmark_size": len(st.session_state.gold), "built_at_utc": st.session_state.built_at,
                    "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_title": "E-Commerce Returns: Who Will Send It Back?",
                    "dataset_url": DATA_URL, "source_filename": st.session_state.source_name,
                    "source_sha256": st.session_state.source_hash, "source_rows": st.session_state.source_rows,
                    "eligible_rows": st.session_state.eligible_rows,
                    "input_usd_per_million_tokens_reference": float(in_rate),
                    "output_usd_per_million_tokens_reference": float(out_rate),
                    "canonical_fields": FIELDS,
                    "protocol_note": "Ground truth is frozen before applying reversible formatting, schema-name aliases and predefined semantic value aliases. Grok is shown canonical field meanings/vocabulary but not the explicit alias lookup.",
                    "limitations": ["Controlled synthetic perturbations, not errors observed in a production feed.", "The frozen baseline is intentionally unchanged from Scenario A; the alias-aware rules arm is included to avoid treating frozen-rule failure as proof of LLM superiority.", "Timing compares local transformations with hosted API response time and is not a matched-hardware benchmark.", "One random seed/sample does not establish broad generalization."],
                    "schema_aliases": SCHEMA_ALIASES, "value_aliases": VALUE_ALIASES,
                },
                "gold_records": st.session_state.gold,
                "scenario_b_inputs": st.session_state.corrupted,
                "frozen_rule_outputs": st.session_state.frozen_outputs,
                "frozen_rule_times_seconds": st.session_state.frozen_times,
                "alias_aware_rule_outputs": st.session_state.hybrid_outputs,
                "alias_aware_rule_times_seconds": st.session_state.hybrid_times,
                "llm_outputs": st.session_state.llm_outputs,
                "llm_model_meta_by_record": st.session_state.model_meta,
                "llm_calls": st.session_state.llm_calls,
            }
            st.download_button("Download Scenario B audit JSON", json.dumps(payload, ensure_ascii=False, indent=2, default=str), file_name="scenario_b_audit.json", mime="application/json")
            rows = []
            for rid, gold in st.session_state.gold.items():
                rows.append({
                    "record_key": rid,
                    "ground_truth_json": json.dumps(gold, ensure_ascii=False),
                    "scenario_b_input_json": json.dumps(st.session_state.corrupted[rid], ensure_ascii=False),
                    "frozen_rules_json": json.dumps(st.session_state.frozen_outputs.get(rid, {}), ensure_ascii=False),
                    "alias_aware_rules_json": json.dumps(st.session_state.hybrid_outputs.get(rid, {}), ensure_ascii=False),
                    "grok_json": json.dumps(st.session_state.llm_outputs.get(rid, {}), ensure_ascii=False),
                    "frozen_rule_seconds": st.session_state.frozen_times.get(rid),
                    "alias_aware_rule_seconds": st.session_state.hybrid_times.get(rid),
                    "grok_call_id": st.session_state.model_meta.get(rid, {}).get("call_id"),
                })
            st.download_button("Download Scenario B record-level CSV", pd.DataFrame(rows).to_csv(index=False), file_name="scenario_b_record_results.csv", mime="text/csv")
            st.warning("Keep audit exports and dataset-derived records private; never commit API keys, train.csv, audit JSON or record-level CSV to a public repository.")

    with t5:
        st.subheader("5 · Scenario B protocol")
        st.markdown("#### Predeclared design")
        protocol = [
            {"Item": "Scenario A linkage", "Specification": "Same source dataset, eligibility rules, canonical fields, baseline numeric/flag/order-ID normalization and reference price settings."},
            {"Item": "New challenge", "Specification": "Alternate header names plus predefined semantic aliases in categorical values."},
            {"Item": "Ground truth", "Specification": "Canonical values saved before the altered headers and values are generated."},
            {"Item": "Arm A", "Specification": "Frozen Scenario A rule parser with no Scenario B alias support."},
            {"Item": "Arm B", "Specification": "Deterministic parser with predeclared schema alias and categorical alias tables."},
            {"Item": "Arm C", "Specification": "Grok with canonical field descriptions and observed canonical vocabularies; no explicit alias-to-value lookup."},
            {"Item": "Metrics", "Specification": "Field and complete-record accuracy, observed wall time, throughput, token use, reference-priced service-equivalent cost and API errors."},
            {"Item": "Current sample sizes", "Specification": "20, 50, 100 or 200 records as actually run; no extrapolation."},
        ]
        st.dataframe(pd.DataFrame(protocol), use_container_width=True, hide_index=True)
        st.markdown("#### Limitations and research integrity")
        st.write("The aliases are controlled and predeclared by the researcher, not collected from real production incidents. Results should be described as performance under this defined Scenario B. The alias-aware rule arm is an important comparison: if a compact lookup table solves the problem accurately and cheaply, that is valuable evidence against unnecessary LLM calls.")
        st.write("For publication, repeat the scenario with multiple predeclared random seeds, report the exact alias lists and prompts, preserve every audit, and do not reuse Scenario A metrics as Scenario B results.")
        st.write("Dataset source: " + DATA_URL + ". Its origin and real-world provenance are not independently verified by this application.")


if __name__ == "__main__":
    main()
