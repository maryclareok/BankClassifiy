# server/app.py
from __future__ import annotations

from pathlib import Path as _Path
from typing import Optional, List, Dict
from uuid import UUID

#from typing import List, Optional
from fastapi import UploadFile, File, Form
import io
import pandas as pd

import io
import os
import re
# Persist user data (models, AllData.csv) under /data for Railway volumes.
import os
from pathlib import Path
import pandas as pd
from dateutil import parser as dateparser
import pdfplumber
from fastapi import FastAPI, Depends, UploadFile, Form
from fastapi.responses import JSONResponse

# --- our auth / db / config / forecast glue ---
from server.config import USER_DATA_ROOT
from server.db import engine
from server.models import Base, User
from server.auth import (
    fastapi_users,
    auth_backend,
    current_active_user,
    UserRead,
    UserCreate,
    UserUpdate,
)
from server.forecast import forecast_next_period

# =============================================================================
# Repo-backed classifier integration (your original logic kept intact)
# =============================================================================
REPO_CLASSIFIER = None
CATEGORIES_PATHS = [
    _Path("categories.txt"),
    _Path(__file__).resolve().parent.parent / "categories.txt",  # project root if running from server/
]

def _load_categories_txt() -> list[str]:
    for p in CATEGORIES_PATHS:
        if p.exists():
            try:
                return [
                    ln.strip()
                    for ln in p.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.strip().startswith("#")
                ]
            except Exception:
                pass
    return []

def _try_make_repo_classifier():
    """
    Returns a callable: (text:str) -> str using the repo if possible,
    else None (caller will use heuristics).
    """
    cat_list = _load_categories_txt()
    if not cat_list:
        return None

    try:
        import importlib
        # Prefer Classify.py
        mod = importlib.import_module("Classify")
    except Exception:
        try:
            mod = importlib.import_module("BankClassify")
        except Exception:
            return None

    # Try common function/class names
    candidate_names = [
        "classify_string", "classify", "classifyText", "classify_text",
        "Classify", "ClassifyLine", "predict_category", "predict"
    ]
    fn = None
    for name in candidate_names:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            break

    if fn is None and hasattr(mod, "Classifier"):
        cls = getattr(mod, "Classifier")
        try:
            inst = cls(categories=cat_list)
            return lambda s: inst.classify(s)
        except Exception:
            pass

    if fn is not None:
        def repo_call(text: str) -> str:
            t = str(text or "").strip()
            if not t:
                return "Other"
            try:
                return str(fn(t))
            except TypeError:
                try:
                    return str(fn(t, cat_list))
                except Exception:
                    return "Other"
            except Exception:
                return "Other"
        return repo_call

    return None

REPO_CLASSIFIER = _try_make_repo_classifier()

# ---- Fallback (simple heuristics) ----
BILL_KEYWORDS = [
    "airtime","data","dstv","gotv","electric","electricity","power","phcn","ikeja electric","eko electric",
    "water","sewage","waste","rent","landlord","estate dues","service charge","cable","netflix","spotify",
    "phone bill","mtn","glo","airtel","9mobile","recharge","subscription",
    "sms charge","vat","maintenance fee","stamp duty","card fee","transfer fee","pos charge","bank charge","charge"
]
EDU_KEYWORDS = [
    "school","tuition","exam","waec","neco","jamb","course","udemy","coursera","textbook","stationery",
    "library","university","polytechnic","fees","education"
]
FUN_KEYWORDS = [
    "restaurant","eatery","food","pizza","kfc","chicken republic","ice cream","bar","club","cinema","movie",
    "entertainment","uber eats","jollof","shawarma","outing","fun","event","concert","games"
]
INCOME_KEYWORDS = [
    "salary","salaries","payroll","wage","stipend","allowance","transfer from","frm","from","credit alert",
    "reversal credit","pos reversal","refund","refunds","deposit","cash deposit","interest","bonus","dividend"
]
BILL_PATTERNS = [re.compile(k, re.IGNORECASE) for k in BILL_KEYWORDS]
EDU_PATTERNS  = [re.compile(k, re.IGNORECASE) for k in EDU_KEYWORDS]
FUN_PATTERNS  = [re.compile(k, re.IGNORECASE) for k in FUN_KEYWORDS]
INC_PATTERNS  = [re.compile(k, re.IGNORECASE) for k in INCOME_KEYWORDS]

def _heuristic_classify(desc: str, amount: float) -> str:
    s = (desc or "").strip()
    if amount is not None and amount > 0: return "Income"
    if any(p.search(s) for p in INC_PATTERNS): return "Income"
    if any(p.search(s) for p in BILL_PATTERNS): return "Bills"
    if any(p.search(s) for p in EDU_PATTERNS):  return "Education"
    if any(p.search(s) for p in FUN_PATTERNS):  return "Fun"
    if s.lower().startswith("period charge"):   return "Bills"
    return "Other"

def classify_guess(desc: str, amount: float) -> str:
    if REPO_CLASSIFIER is not None:
        try:
            cat = REPO_CLASSIFIER(desc)
            if cat and isinstance(cat, str):
                return cat
        except Exception:
            pass
    return _heuristic_classify(desc, amount)

def add_guesses(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["guess"] = [classify_guess(d, a) for d, a in zip(out["desc"], out["amount"])]
    return out

# =============================================================================
# App & Auth routes
# =============================================================================
app = FastAPI(title="BankClassify API (Auth + PDF + Forecast)")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "*"  # loosened for dev; restrict in prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    # create tables (users)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# FastAPI Users (JWT login/register/users)
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

# =============================================================================
# Storage / constants
# =============================================================================


DATA_ROOT = Path(os.getenv("USER_DATA_ROOT", "/data/models")).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

MONEY_RE = r"\(?[₦$€£]?\s?\d[\d,]*\.?\d{0,2}\)?(?:\s*(?:DR|CR))?"
DATE_RE  = r"(\d{1,2}[-/ ]?[A-Za-z]{3,9}[-/ ]?\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})"

HEADER_DROP_PATTERNS = [
    r"bank\s+statement",
    r"statement\s+period",
    r"account\s+summary",
    r"opening\s+balance",
    r"closing\s+balance",
    r"page\s+\d+",
    r"customer\s+copy",
    r"confidential",
    r"africa'?s\s+global\s+bank",
]

MONTH_WORD = r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUNE|JUL|JULY|AUG|SEP|SEPT|OCT|NOV|DEC)"
ORDINAL    = r"\d{1,2}(?:ST|ND|RD|TH)?"
PERIOD_ONLY = re.compile(
    rf"^\s*-?\s*(?:{ORDINAL}\s*-\s*{MONTH_WORD}\s*{ORDINAL}|{MONTH_WORD}\s*{ORDINAL}(?:\s*-\s*{MONTH_WORD}\s*{ORDINAL})?)\s*$",
    re.IGNORECASE,
)

# =============================================================================
# Helpers: amounts & dates & text cleanup
# =============================================================================
def parse_amount(s: str) -> Optional[float]:
    if s is None: return None
    t = str(s).strip()
    if not t: return None
    t_clean = re.sub(r"[₦$€£,\s]", "", t)
    neg = False
    if t_clean.upper().endswith("DR"):
        neg = True; t_clean = t_clean[:-2]
    elif t_clean.upper().endswith("CR"):
        t_clean = t_clean[:-2]
    if re.match(r"^\(.*\)$", t):
        neg = True; t_clean = t_clean.strip("()")
    try:
        v = float(t_clean)
        return -v if neg else v
    except ValueError:
        return None

def smart_parse_date(s: str):
    if not s or not str(s).strip(): return None
    try:
        dt = dateparser.parse(str(s), dayfirst=True, fuzzy=True)
        return pd.to_datetime(dt).date() if dt else None
    except Exception:
        return None

def strip_date_tokens(text: str) -> str:
    if not text: return ""
    text = re.sub(r"^\s*"+DATE_RE+r"\s+", "", text)
    text = re.sub(r"\b(\d{1,2}[-/ ]?[A-Za-z]{3,9}[-/ ]?\d{2,4})\b", "", text)
    return re.sub(r"\s+", " ", text).strip()

def strip_trailing_money_tokens(text: str) -> str:
    if not text: return ""
    prev = None; cur = text
    while prev != cur:
        prev = cur
        cur = re.sub(r"\s*"+MONEY_RE+r"\s*$", "", cur).rstrip()
    return cur

def normalize_periodish_desc(desc: str) -> str:
    if not desc:
        return desc
    s = desc.strip()
    s = re.sub(r"^\s*-\s*", "", s)  # "- JULY 10TH" -> "JULY 10TH"
    if PERIOD_ONLY.match(s):
        return f"Period charge ({s})"
    return desc

# =============================================================================
# PDF geometry parser (your original, preserved)
# =============================================================================
def parse_pdf_transactions(pdf_bytes: bytes) -> pd.DataFrame:
    date_full  = re.compile(r"^\s*"+DATE_RE+r"\s*$")
    money_full = re.compile(r"^"+MONEY_RE+r"$", re.IGNORECASE)

    def is_date(tok: str) -> bool:
        return bool(date_full.match(tok))

    def is_money(tok: str) -> bool:
        t = tok.strip()
        if not money_full.match(t):
            return False
        val = parse_amount(t)
        if val is None:
            return False
        if re.search(r"[.,]", t) or re.search(r"(?:DR|CR)", t, re.IGNORECASE) or "(" in t or ")" in t:
            return True
        return abs(val) >= 100  # filter accidental '1', '2', etc.

    def looks_header(desc: str) -> bool:
        s = (desc or "").lower().strip()
        if not s: return False
        for pat in HEADER_DROP_PATTERNS:
            if re.search(pat, s):
                return True
        if len(s) < 3:
            return True
        if re.fullmatch(r"[-–—•\s]+", s):
            return True
        return False

    def cluster_rows(words: List[Dict], y_tol: float = 3.2):
        rows: List[List[Dict]] = []
        for w in words:
            ymid = (w["top"] + w["bottom"]) / 2.0
            placed = False
            for row in rows:
                rymid = row[0]["_ymid"]
                if abs(ymid - rymid) <= y_tol:
                    row.append({**w, "_ymid": ymid}); placed = True; break
            if not placed:
                rows.append([{**w, "_ymid": ymid}])
        rows.sort(key=lambda r: r[0]["_ymid"])
        for r in rows: r.sort(key=lambda w: w["x0"])
        return rows

    def compute_amount_and_money_indices(tokens: List[Dict]):
        money_idx = [i for i,t in enumerate(tokens) if is_money(t["text"])]
        if not money_idx: return None, set()
        money_idx.sort(key=lambda i: tokens[i]["x0"])
        keep = money_idx[-3:]  # up to debit, credit, balance
        keep.sort(key=lambda i: tokens[i]["x0"])
        if len(keep) >= 2:
            deb_txt = tokens[keep[-3]]["text"] if len(keep) == 3 else tokens[keep[-2]]["text"]
            cre_txt = tokens[keep[-2]]["text"] if len(keep) == 3 else tokens[keep[-1]]["text"]
            dval = parse_amount(deb_txt) or 0.0
            cval = parse_amount(cre_txt) or 0.0
            amount = cval - dval
        else:
            amount = parse_amount(tokens[keep[-1]]["text"]) or 0.0
        return amount, set(keep)

    rows_out = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=1.0,
                y_tolerance=3.0,
                keep_blank_chars=False,
                use_text_flow=True
            )
            rows = cluster_rows(words, y_tol=3.2)
            for toks in rows:
                # leftmost date in first few tokens
                date_i = None
                for i, t in enumerate(toks[:6]):
                    if is_date(t["text"]):
                        date_i = i; break
                if date_i is None:
                    continue
                date_str = toks[date_i]["text"].strip()
                if not smart_parse_date(date_str):
                    continue

                amount, money_set = compute_amount_and_money_indices(toks)
                if amount is None:
                    continue

                # Description = all non-date, non-money tokens
                desc_tokens = [
                    t for i,t in enumerate(toks)
                    if i != date_i and i not in money_set and not is_date(t["text"]) and not is_money(t["text"])
                ]
                desc = " ".join(t["text"] for t in desc_tokens)
                desc = strip_trailing_money_tokens(strip_date_tokens(desc))
                desc = re.sub(r"\s+", " ", desc).strip()

                if not desc:
                    # fallback: slice between date and first/rightmost money x-range
                    last_money_x = max((toks[i]["x0"] for i in money_set), default=toks[-1]["x0"])
                    date_x1 = toks[date_i]["x1"]
                    mid = [
                        t for i,t in enumerate(toks)
                        if (t["x0"] >= date_x1 - 1.0 and t["x1"] <= last_money_x + 0.5 and i not in money_set and i != date_i)
                    ]
                    desc = " ".join(t["text"] for t in mid)
                    desc = strip_trailing_money_tokens(strip_date_tokens(desc))
                    desc = re.sub(r"\s+", " ", desc).strip()

                if not desc or looks_header(desc):
                    continue

                desc = normalize_periodish_desc(desc)
                rows_out.append({"date": date_str, "desc": desc, "amount": amount})

    if rows_out:
        df = pd.DataFrame(rows_out)
        df["date"] = df["date"].apply(smart_parse_date)
        df = df.dropna(subset=["date", "desc"])
        return df

    # ------- Fallback via tables (rare) -------
    rows_tb = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf2:
        for page in pdf2.pages:
            tables = page.extract_tables() or []
            for tb in tables:
                if not tb or len(tb) < 2: continue
                header = [(h.strip() if isinstance(h, str) else "") for h in tb[0]]
                body   = tb[1:]
                idx = {"date": None, "desc": None, "debit": None, "credit": None, "amount": None}
                for i,h in enumerate(header):
                    hl = h.lower()
                    if "date" in hl and idx["date"] is None: idx["date"] = i
                    elif any(k in hl for k in ["description","details","narration","particular"]): idx["desc"]=i
                    elif hl=="debit": idx["debit"]=i
                    elif hl=="credit": idx["credit"]=i
                    elif "amount" in hl: idx["amount"]=i
                if idx["date"] is None or all(idx[k] is None for k in ["amount","debit","credit"]):
                    continue
                for r in body:
                    if not r or all(x in (None,"") for x in r): continue
                    def get(i): return (r[i] if i is not None and i < len(r) else "") or ""
                    dstr = str(get(idx["date"])).strip()
                    if not smart_parse_date(dstr): continue
                    if idx["amount"] is not None:
                        aval = parse_amount(str(get(idx["amount"])))
                    else:
                        dval = parse_amount(str(get(idx["debit"]))) if idx["debit"] is not None else None
                        cval = parse_amount(str(get(idx["credit"]))) if idx["credit"] is not None else None
                        aval = (0 if cval is None else cval) - (0 if dval is None else dval)
                    if aval is None: continue
                    desc = str(get(idx["desc"])).strip() if idx["desc"] is not None else ""
                    desc = strip_trailing_money_tokens(strip_date_tokens(desc))
                    if not desc or any(re.search(p, desc.lower()) for p in HEADER_DROP_PATTERNS):
                        continue
                    desc = normalize_periodish_desc(desc)
                    rows_tb.append({"date": dstr, "desc": desc, "amount": aval})

        if rows_tb:
            df = pd.DataFrame(rows_tb)
            df["date"] = df["date"].apply(smart_parse_date)
            df = df.dropna(subset=["date", "desc"])
            return df

    return pd.DataFrame(columns=["date","desc","amount"])

# =============================================================================
# CSV canonical normalization (when user uploads CSV instead of PDF)
# =============================================================================
def normalize_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df_raw.columns:
        s = str(c).strip().lower()
        if "date" in s and "update" not in s: rename[c] = "date"
        elif any(k in s for k in ["desc","narration","details","particular"]): rename[c] = "desc"
        elif s == "debit": rename[c] = "Debit"
        elif s == "credit": rename[c] = "Credit"
        elif "amount" in s and "net" not in s: rename[c] = "amount"
        elif "balance" in s: rename[c] = "Balance"
    df = df_raw.rename(columns=rename).copy()

    if "amount" not in df.columns and ("Debit" in df.columns or "Credit" in df.columns):
        deb = df.get("Debit", pd.Series([0]*len(df))).apply(parse_amount)
        cred = df.get("Credit", pd.Series([0]*len(df))).apply(parse_amount)
        df["amount"] = (cred.fillna(0) - deb.fillna(0))

    for need in ("date","desc","amount"):
        if need not in df.columns: df[need] = None

    df["date"]   = df["date"].apply(smart_parse_date)
    df["amount"] = df["amount"].apply(parse_amount)
    df["desc"]   = (
        df["desc"].astype(str)
        .apply(strip_trailing_money_tokens)
        .apply(strip_date_tokens)
        .apply(normalize_periodish_desc)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    out = df[["date","desc","amount"]].dropna(subset=["date","desc","amount"])

    # drop numeric-only desc or zero-amount artifacts
    mask_numeric_desc = out["desc"].str.fullmatch(r"[0\s.,()DRCR-]+", case=False, na=False)
    mask_zero_amt     = out["amount"].fillna(0).eq(0)
    out = out[~(mask_numeric_desc | mask_zero_amt)]
    return out

# =============================================================================
# User storage helpers
# =============================================================================
def user_paths_for(uid: str):
    udir = DATA_ROOT / uid
    udir.mkdir(parents=True, exist_ok=True)
    return {"root": udir, "all_data": udir / "AllData.csv", "categories": udir / "categories.txt"}

# =============================================================================
# API
# =============================================================================
@app.get("/")
def health():
    return {"ok": True}

# ...keep the rest of your imports and helpers...

@app.post("/classify/upload")
async def classify_upload(
    user_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Accepts one or many statements (PDF/CSV). We extract & merge all rows,
    then return a single preview with guesses.
    """
    # Collect the upload list
    uploads: List[UploadFile] = []
    if file is not None:
        uploads.append(file)
    if files:
        uploads.extend(files)

    if not uploads:
        return JSONResponse({"error": "No files provided. Use 'file' or 'files[]'."}, status_code=400)

    frames: List[pd.DataFrame] = []

    for up in uploads:
        name = (up.filename or "").lower()
        content = await up.read()

        try:
            if name.endswith(".pdf") or (up.content_type or "").lower() == "application/pdf":
                # Primary path: your pdfplumber geometry-aware extraction
                df_part = parse_pdf_transactions(content)
                # If you later add monopoly-core, you can try it first and fallback to pdfplumber.
            elif name.endswith(".csv"):
                # Robust CSV read (uses your canonical normalizer)
                df_part = normalize_to_canonical(pd.read_csv(io.BytesIO(content)))
            else:
                # Default to CSV attempt if extension missing/unknown
                df_part = normalize_to_canonical(pd.read_csv(io.BytesIO(content)))
        except Exception:
            # Skip bad files but continue others
            continue

        if df_part is not None and not df_part.empty:
            frames.append(df_part)

    if not frames:
        return JSONResponse({"error": "No transactions found across uploaded files."}, status_code=422)

    # Merge, drop duplicates
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "desc", "amount"]).reset_index(drop=True)

    # Add guesses (repo-backed or heuristics)
    out = add_guesses(df)

    # Show a small “needs-review” batch: anything that wasn’t clearly classified
    needs = out[out["guess"].isin(["", "Other"])].head(20).to_dict(orient="records")

    return {
        "preview": out.head(100).to_dict(orient="records"),
        "review_batch": needs,
        "rows": int(len(out)),
        "files_processed": len(frames)
    }

@app.post("/classify/correct")
async def classify_correct(
    labeled_csv: UploadFile | None = None,
    user: User = Depends(current_active_user),
):
    if not labeled_csv:
        return JSONResponse({"error": "labeled_csv missing"}, status_code=400)

    try:
        df_labeled = pd.read_csv(labeled_csv.file)
    except Exception as e:
        return JSONResponse({"error": f"Failed to read CSV: {e}"}, status_code=400)

    # Accept either your earlier 'final_label' or a ready 'cat' column
    cols_lower = [c.lower() for c in df_labeled.columns]
    if "final_label" in cols_lower and "cat" not in cols_lower:
        df_labeled.columns = [c.lower() for c in df_labeled.columns]
        df_save = df_labeled[["date","desc","amount","final_label"]].rename(columns={"final_label":"cat"})
    else:
        df_labeled.columns = [c.lower() for c in df_labeled.columns]
        if not {"date","desc","amount","cat"}.issubset(df_labeled.columns):
            return JSONResponse({"error":"CSV must contain: date, desc, amount, cat (or final_label)"}, status_code=400)
        df_save = df_labeled[["date","desc","amount","cat"]].copy()

    df_save["date"] = pd.to_datetime(df_save["date"], errors="coerce").dt.date
    df_save = df_save.dropna(subset=["date","desc"])

    p = user_paths_for(str(user.id))
    past = pd.read_csv(p["all_data"]) if p["all_data"].exists() else pd.DataFrame(columns=["date","desc","amount","cat"])
    past.columns = [c.lower() for c in past.columns]
    for c in ["date","desc","amount","cat"]:
        if c not in past.columns:
            past[c] = pd.NA
    past["date"] = pd.to_datetime(past["date"], errors="coerce").dt.date

    merged = pd.concat([past[["date","desc","amount","cat"]], df_save], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date","desc","amount"], keep="last")
    merged.to_csv(p["all_data"], index=False)
    return {"status":"ok","stored_rows": int(len(merged))}

# Forecast using user's AllData.csv
@app.post("/forecast/run")
def run_forecast(
    horizon: str = Form("next_month"),
    user: User = Depends(current_active_user),
):
    """
    horizon: "next_week" or "next_month"
    Uses models/<user_id>/AllData.csv with the labels you've stored via /classify/correct.
    """
    res = forecast_next_period(user_id=str(user.id), horizon=horizon)
    return {
        "as_of": str(res.as_of),
        "horizon": res.horizon,
        "by_category": res.by_category,
        "total": res.total,
    }
