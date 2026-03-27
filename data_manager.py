"""
data_manager.py — Persistent Master Data Store
Warehouse Rental Management Analytics — Data Analytics MGB Project

Stores all master data in JSON files that persist on Streamlit Cloud:
  - business_config.json  → your name, phone, email, bank details
  - owners.json           → family owner records
  - warehouses.json       → warehouse properties
  - tenants.json          → tenant contacts
  - rentals.json          → active lease agreements

All data is managed through the dashboard UI — no CSV/code editing needed.
The transaction dataset (warehouse_data.csv) is rebuilt from these masters
whenever data changes.
"""

import json
import os
import uuid
from datetime import datetime, date
import pandas as pd
import numpy as np
import streamlit as st

# ── File paths (same folder as app.py) ────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

BUSINESS_FILE  = os.path.join(DATA_DIR, "business_config.json")
OWNERS_FILE    = os.path.join(DATA_DIR, "owners.json")
WAREHOUSES_FILE= os.path.join(DATA_DIR, "warehouses.json")
TENANTS_FILE   = os.path.join(DATA_DIR, "tenants.json")
RENTALS_FILE   = os.path.join(DATA_DIR, "rentals.json")


# ════════════════════════════════════════════════════════════════════════════
# DEFAULT SEED DATA  (used only if JSON files don't exist yet)
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_BUSINESS = {
    "business_name":    "WareHub Family Properties",
    "owner_name":       "Rajesh Shah",
    "phone":            "9571789145",
    "email":            "",
    "address":          "123, Industrial Estate, Udaipur, Rajasthan — 313001",
    "gstin":            "08AABCW1234F1ZX",
    "pan":              "AABCW1234F",
    "bank_name":        "State Bank of India",
    "bank_account":     "32145678901234",
    "bank_ifsc":        "SBIN0001234",
    "bank_branch":      "Udaipur Main Branch",
    "upi_id":           "warehub@sbi",
    "gst_rate":         18,
    "due_days":         10,
    "penalty_pct":      2,
}

DEFAULT_OWNERS = [
    {"id": "OWN-01", "name": "Rajesh Shah",  "phone": "9876543210", "email": "rajesh.shah@warehub.in",  "city": "Udaipur"},
    {"id": "OWN-02", "name": "Meena Joshi",  "phone": "9823456789", "email": "meena.joshi@warehub.in",  "city": "Jaipur"},
    {"id": "OWN-03", "name": "Amit Patel",   "phone": "9812345678", "email": "amit.patel@warehub.in",   "city": "Ahmedabad"},
    {"id": "OWN-04", "name": "Sunita Gupta", "phone": "9898765432", "email": "sunita.gupta@warehub.in", "city": "Surat"},
]

DEFAULT_WAREHOUSES = [
    {"id":"WH-01","name":"Udaipur Cold Hub",    "location":"Udaipur",   "state":"Rajasthan",  "type":"Cold Storage",    "size":"Medium","owner_id":"OWN-01","base_rent":32000,"capacity_mt":120,"area_sqft":8000},
    {"id":"WH-02","name":"Jaipur Dry Store",    "location":"Jaipur",    "state":"Rajasthan",  "type":"Dry Warehouse",   "size":"Large", "owner_id":"OWN-02","base_rent":45000,"capacity_mt":200,"area_sqft":12000},
    {"id":"WH-03","name":"Ahmedabad Mega Hub",  "location":"Ahmedabad", "state":"Gujarat",    "type":"Distribution Hub","size":"Large", "owner_id":"OWN-03","base_rent":72000,"capacity_mt":350,"area_sqft":20000},
    {"id":"WH-04","name":"Surat Textile WH",    "location":"Surat",     "state":"Gujarat",    "type":"Dry Warehouse",   "size":"Medium","owner_id":"OWN-04","base_rent":28000,"capacity_mt":160,"area_sqft":9500},
    {"id":"WH-05","name":"Jaipur Pharma Store", "location":"Jaipur",    "state":"Rajasthan",  "type":"Cold Storage",    "size":"Small", "owner_id":"OWN-02","base_rent":22000,"capacity_mt":80, "area_sqft":6000},
    {"id":"WH-06","name":"Udaipur Dist Hub",    "location":"Udaipur",   "state":"Rajasthan",  "type":"Distribution Hub","size":"Large", "owner_id":"OWN-01","base_rent":65000,"capacity_mt":260,"area_sqft":15000},
]

DEFAULT_TENANTS = [
    {"id":"TNT-01","name":"FreshMart Retail",   "type":"Business",  "industry":"Retail",     "contact":"Suresh Kumar","phone":"9871234560","email":"procurement@freshmart.in",  "tenure_months":36,"wh_id":"WH-01"},
    {"id":"TNT-02","name":"GrainCorp Ltd",      "type":"Business",  "industry":"Agriculture","contact":"Priya Sharma","phone":"9812345670","email":"accounts@graincorp.co.in",  "tenure_months":28,"wh_id":"WH-02"},
    {"id":"TNT-03","name":"QuickShip Logistics","type":"Business",  "industry":"Logistics",  "contact":"Vikram Singh","phone":"9834567890","email":"finance@quickship.in",       "tenure_months":42,"wh_id":"WH-03"},
    {"id":"TNT-04","name":"TextilePro Exports", "type":"Business",  "industry":"Textile",    "contact":"Anita Desai", "phone":"9845678901","email":"admin@textilepro.com",      "tenure_months":18,"wh_id":"WH-04"},
    {"id":"TNT-05","name":"PharmaPlus India",   "type":"Business",  "industry":"Pharma",     "contact":"Dr. R. Mehta","phone":"9856789012","email":"ap@pharmaplus.in",           "tenure_months":30,"wh_id":"WH-05"},
    {"id":"TNT-06","name":"Metro Distributors", "type":"Business",  "industry":"FMCG",       "contact":"Rajan Gupta", "phone":"9867890123","email":"accounts@metrodist.co.in",  "tenure_months":24,"wh_id":"WH-06"},
]

DEFAULT_RENTALS = [
    {"id":"RNT-01","tenant_id":"TNT-01","wh_id":"WH-01","start_date":"2023-03-01","end_date":"2025-02-28","monthly_rent":85000,"status":"Active"},
    {"id":"RNT-02","tenant_id":"TNT-02","wh_id":"WH-02","start_date":"2022-07-01","end_date":"2024-06-30","monthly_rent":72000,"status":"Active"},
    {"id":"RNT-03","tenant_id":"TNT-03","wh_id":"WH-03","start_date":"2024-01-01","end_date":"2026-12-31","monthly_rent":145000,"status":"Active"},
    {"id":"RNT-04","tenant_id":"TNT-04","wh_id":"WH-04","start_date":"2023-09-01","end_date":"2025-08-31","monthly_rent":58000,"status":"Active"},
    {"id":"RNT-05","tenant_id":"TNT-05","wh_id":"WH-05","start_date":"2024-03-01","end_date":"2025-02-28","monthly_rent":65000,"status":"Active"},
    {"id":"RNT-06","tenant_id":"TNT-06","wh_id":"WH-06","start_date":"2023-11-01","end_date":"2025-10-31","monthly_rent":110000,"status":"Active"},
]


# ════════════════════════════════════════════════════════════════════════════
# JSON I/O HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _read(path: str, default) -> list | dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default if default is not None else []


def _write(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Read
# ════════════════════════════════════════════════════════════════════════════

def get_business() -> dict:
    return _read(BUSINESS_FILE, DEFAULT_BUSINESS)

def get_owners() -> list:
    return _read(OWNERS_FILE, list(DEFAULT_OWNERS))

def get_warehouses() -> list:
    return _read(WAREHOUSES_FILE, list(DEFAULT_WAREHOUSES))

def get_tenants() -> list:
    return _read(TENANTS_FILE, list(DEFAULT_TENANTS))

def get_rentals() -> list:
    return _read(RENTALS_FILE, list(DEFAULT_RENTALS))


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Write
# ════════════════════════════════════════════════════════════════════════════

def save_business(data: dict) -> None:
    _write(BUSINESS_FILE, data)
    st.cache_data.clear()

def save_owners(data: list) -> None:
    _write(OWNERS_FILE, data)
    st.cache_data.clear()

def save_warehouses(data: list) -> None:
    _write(WAREHOUSES_FILE, data)
    st.cache_data.clear()

def save_tenants(data: list) -> None:
    _write(TENANTS_FILE, data)
    st.cache_data.clear()

def save_rentals(data: list) -> None:
    _write(RENTALS_FILE, data)
    st.cache_data.clear()


# ════════════════════════════════════════════════════════════════════════════
# TRANSACTION BUILDER — generate analytics dataset from master data
# ════════════════════════════════════════════════════════════════════════════

def build_transaction_df() -> pd.DataFrame:
    """
    Build the full analytics-ready transaction DataFrame from master JSON data.
    Called whenever master data changes or app loads.
    """
    import numpy as np
    from datetime import timedelta

    business   = get_business()
    owners_l   = get_owners()
    warehouses_l = get_warehouses()
    tenants_l  = get_tenants()
    rentals_l  = get_rentals()

    if not rentals_l or not tenants_l:
        return pd.DataFrame()

    owner_map = {o["id"]: o for o in owners_l}
    wh_map    = {w["id"]: w for w in warehouses_l}
    tnt_map   = {t["id"]: t for t in tenants_l}

    gst_rate = business.get("gst_rate", 18) / 100

    size_map  = {"Small": 1, "Medium": 2, "Large": 3}
    type_map  = {"General": 1, "Dry Warehouse": 2, "Cold Storage": 3, "Distribution Hub": 4}
    ttype_map = {"Individual": 0, "Business": 1}

    rows = []
    row_num = 1

    for rental in rentals_l:
        tnt_id = rental.get("tenant_id", "")
        wh_id  = rental.get("wh_id", "")

        tnt = tnt_map.get(tnt_id)
        wh  = wh_map.get(wh_id)
        if not tnt or not wh:
            continue

        own  = owner_map.get(wh.get("owner_id", ""), {})
        rent = float(rental.get("monthly_rent", 0))
        gst  = round(rent * gst_rate, 0)
        total = rent + gst

        # Generate monthly records for the rental period
        try:
            start = pd.Timestamp(rental.get("start_date", "2023-01-01"))
            end   = pd.Timestamp(rental.get("end_date",   "2024-12-31"))
        except Exception:
            continue

        months = pd.date_range(start, end, freq="MS")

        # Behaviour profile from tenant tenure
        tenure = int(tnt.get("tenure_months", 12))
        if tenure >= 36:
            profile = "on_time"
        elif tenure >= 18:
            profile = "delayed"
        else:
            profile = "risky"

        np.random.seed(hash(tnt_id + wh_id) % (2**31))

        for mo in months:
            if mo > pd.Timestamp.today():
                break

            invoice_date = mo
            due_date     = mo + pd.Timedelta(days=business.get("due_days", 10))

            # Payment logic
            if profile == "on_time":
                paid = np.random.choice([True, False], p=[0.92, 0.08])
                delay = 0 if paid and np.random.random() > 0.3 else np.random.randint(1, 8) if paid else 0
            elif profile == "delayed":
                paid = np.random.choice([True, False], p=[0.78, 0.22])
                delay = np.random.randint(5, 25) if paid else 0
            else:
                paid = np.random.choice([True, False], p=[0.55, 0.45])
                delay = np.random.randint(10, 40) if paid else 0

            if paid:
                pay_date   = due_date + pd.Timedelta(days=delay)
                pay_status = "Paid"
                amt_paid   = int(total)
                balance    = 0
                if delay == 0:
                    beh = "On-time"
                elif delay <= 10:
                    beh = "Slightly Delayed"
                else:
                    beh = "Delayed"
            else:
                pay_date   = None
                pay_status = "Not Paid"
                amt_paid   = 0
                balance    = int(total)
                beh        = "Defaulted"
                days_outstanding = (pd.Timestamp.today() - due_date).days
                delay = max(0, int(days_outstanding))

            reminder = "Yes" if (not paid or delay > 10) else "No"

            lease_dur = max(1, int((pd.Timestamp(rental.get("end_date","2024-12-31")) -
                                    pd.Timestamp(rental.get("start_date","2023-01-01"))).days / 30))

            risk = round(
                (min(delay, 90) / 90 * 40) +
                ({"On-time":1,"Slightly Delayed":2,"Delayed":3,"Defaulted":4}.get(beh,1) / 4 * 40) +
                ((1 - min(tenure, 48) / 48) * 20), 1
            )

            if beh in ["On-time","Slightly Delayed"] and rent >= 50000:
                seg = "High-Value Regular"
            elif beh == "On-time":
                seg = "Regular Payer"
            elif beh in ["Delayed","Slightly Delayed"]:
                seg = "Late Payer"
            else:
                seg = "Defaulter"

            rows.append({
                "Row_ID":                 f"TXN-{row_num:04d}",
                "Tenant_ID":              tnt_id,
                "Tenant_Name":            tnt.get("name",""),
                "Owner_ID":               wh.get("owner_id",""),
                "Owner_Name":             own.get("name",""),
                "WH_ID":                  wh_id,
                "Warehouse_Location":     wh.get("location",""),
                "Warehouse_State":        wh.get("state",""),
                "Warehouse_Type":         wh.get("type",""),
                "Warehouse_Size":         wh.get("size",""),
                "Month":                  mo.strftime("%Y-%m-%d"),
                "Month_Name":             mo.strftime("%b-%Y"),
                "Quarter":               f"Q{((mo.month-1)//3)+1}-{mo.year}",
                "Monthly_Rent_INR":       int(rent),
                "GST_18pct_INR":          int(gst),
                "Total_Invoice_INR":      int(total),
                "Lease_Duration_Months":  lease_dur,
                "Payment_Status":         pay_status,
                "Payment_Date":           pay_date.strftime("%Y-%m-%d") if pay_date else "",
                "Delay_Days":             int(delay),
                "Amount_Paid_INR":        amt_paid,
                "Balance_Due_INR":        balance,
                "Invoice_Sent":           "Yes",
                "Invoice_Date":           invoice_date.strftime("%Y-%m-%d"),
                "Due_Date":               due_date.strftime("%Y-%m-%d"),
                "Reminder_Sent":          reminder,
                "Email_Sent":             "Yes",
                "WhatsApp_Sent":          "Yes",
                "Tenant_Type":            tnt.get("type","Business"),
                "Industry_Type":          tnt.get("industry",""),
                "Payment_Behavior":       beh,
                "Customer_Tenure_Months": tenure,
                "Is_Paid_Binary":         1 if paid else 0,
                "Is_Delayed_Binary":      1 if (paid and delay > 0) else 0,
                "Revenue_Collected_INR":  amt_paid,
                "Size_Encoded":           size_map.get(wh.get("size","Medium"), 2),
                "Type_Encoded":           type_map.get(wh.get("type","Dry Warehouse"), 2),
                "Behavior_Encoded":       {"On-time":1,"Slightly Delayed":2,"Delayed":3,"Defaulted":4}.get(beh,1),
                "TenantType_Encoded":     ttype_map.get(tnt.get("type","Business"), 1),
                "Risk_Score":             risk,
                "Tenant_Segment":         seg,
                "Month_Num":              (mo.year - 2023) * 12 + mo.month,
                "KMeans_Cluster":         0,
                "Cluster_Label":          seg,
                "Tenant_Email":           tnt.get("email",""),
                "Tenant_Phone":           tnt.get("phone",""),
                "Contact_Person":         tnt.get("contact",""),
                "Owner_Email":            own.get("email",""),
                "Owner_Phone":            own.get("phone",""),
            })
            row_num += 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Month"] = pd.to_datetime(df["Month"])
    return df


@st.cache_data
def get_transaction_df() -> pd.DataFrame:
    """Cached version — cleared whenever master data changes via save_*()."""
    df = build_transaction_df()
    if df.empty:
        # Fallback to CSV if JSON masters haven't been set up yet
        csv_path = os.path.join(DATA_DIR, "warehouse_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    return df


# ════════════════════════════════════════════════════════════════════════════
# NEXT ID GENERATORS
# ════════════════════════════════════════════════════════════════════════════

def _next_id(records: list, prefix: str) -> str:
    nums = []
    for r in records:
        try:
            n = int(r.get("id","").replace(prefix+"-",""))
            nums.append(n)
        except Exception:
            pass
    next_n = max(nums) + 1 if nums else 1
    return f"{prefix}-{next_n:02d}"
