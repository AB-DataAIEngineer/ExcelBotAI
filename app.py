"""
Airline Data Chatbot - Backend
-------------------------------
User English mein sawaal poochta hai -> Gemini AI SQL query banata hai ->
query Excel data pe chalti hai -> jawab wapas JSON mein jata hai.

Data Excel file se padha jata hai aur memory mein ek SQLite table
("untitled") mein load hota hai. Kisi online database ki zaroorat nahi.
"""

import json
import os
import re
import sqlite3
import threading
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Settings (.env file se)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# "gemini-flash-latest" hamesha Google ka latest Flash model use karta hai
AI_MODEL = os.getenv("GEMINI_MODEL", "").strip() or "gemini-flash-latest"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EXCEL_FILE = os.getenv("EXCEL_FILE", "").strip()
EXCEL_SHEET = os.getenv("EXCEL_SHEET", "").strip()
TABLE_NAME = "untitled"

if not GEMINI_API_KEY:
    raise SystemExit(
        "GEMINI_API_KEY nahi mili. Folder mein .env file banao aur usme "
        "GEMINI_API_KEY=apni-key likho (Google AI Studio se)."
    )

# OpenAI library hi use ho rahi hai, bas requests Google Gemini ko ja rahi hain
client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


# ---------------------------------------------------------------------------
# 2. Excel file load karna
# ---------------------------------------------------------------------------
def find_excel_file() -> Path:
    """.env mein diya naam use karo, warna folder ki pehli Excel file."""
    if EXCEL_FILE:
        path = Path(EXCEL_FILE)
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.exists():
            raise SystemExit(f"Excel file nahi mili: {path}")
        return path

    candidates = [
        p for p in sorted(BASE_DIR.iterdir())
        if p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith("~$")
    ]
    if not candidates:
        raise SystemExit(
            f"{BASE_DIR} folder mein koi Excel (.xlsx) ya CSV (.csv) file nahi mili. "
            "File isi folder mein rakho ya .env mein EXCEL_FILE set karo."
        )
    return candidates[0]


def clean_column_name(name) -> str:
    """'Premium/Full Economy Class Pax' -> 'Premium_Full_Economy_Class_Pax'"""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
    return cleaned or "column"


def load_data() -> pd.DataFrame:
    path = find_excel_file()
    print(f"Data file load ho rahi hai: {path.name}")
    if path.suffix.lower() == ".csv":
        # CSV alag-alag encoding mein save ho sakti hai, isliye ek-ek karke try karo
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
    else:
        df = pd.read_excel(path, sheet_name=EXCEL_SHEET or 0)

    # Column names SQL-friendly banao (spaces, '/', '-' hata ke '_')
    new_cols, seen = [], {}
    for col in df.columns:
        c = clean_column_name(col)
        if c in seen:  # duplicate names se bachne ke liye
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        new_cols.append(c)
    df.columns = new_cols

    # Dates ko simple text (YYYY-MM-DD) bana do taaki SQL mein aasani rahe
    for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")

    print(f"{len(df)} rows aur {len(df.columns)} columns load hue.")
    return df


df_data = load_data()

# In-memory SQLite database banao aur data daal do
db = sqlite3.connect(":memory:", check_same_thread=False)
df_data.to_sql(TABLE_NAME, db, index=False)
db_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 3. AI ke liye prompt
# ---------------------------------------------------------------------------
COLUMN_HINTS = """
Ticket_Purchase_Month_YYYY_MM: month the ticket was purchased.
Trip_Month_YYYY_MM: month the trip took place.
Dominant_Operating_Airline: airline operating the main part of the trip.
Operating_Airline_1 .. Operating_Airline_6: airline for each segment ('0' or blank = no segment).
Origin / Destination: airport codes where the trip starts / ends.
Trip_Origin_City, Trip_Origin_Country, Trip_Origin_Region: where the trip starts.
Trip_Destination_City, Trip_Destination_Country, Trip_Destination_Region: where the trip ends.
Stop_1 .. Stop_6: stopover airports ('0' or blank = no stop).
<Class>_Pax: number of passengers in that class.
<Class>_Blended_Payment_Amount: average/blended payment for that class.
<Class>_Actual_Payment_Amount: actual payment received for that class.
Classes are: Business_Class, Discounted_Economy_Class, First_Class, Other_Classes, Premium_Full_Economy_Class.
Total_Pax, Total_Blended_Payment_Amount, Total_Actual_Payment_Amount: totals across all classes.

Airline codes: AI=Air India, UA=United Airlines, AC=Air Canada, LH=Lufthansa,
EK=Emirates, BA=British Airways, QR=Qatar Airways, QF=Qantas, AF=Air France,
KL=KLM Royal Dutch Airlines, VS=Virgin Atlantic.
Airport codes: DEL=Delhi, BOM=Mumbai, BLR=Bengaluru, MAA=Chennai, HYD=Hyderabad,
CCU=Kolkata, LHR=London, EWR/JFK=New York, ORD=Chicago, SFO=San Francisco,
YYZ=Toronto, FRA=Frankfurt, DXB=Dubai, CDG=Paris, SYD=Sydney.
"""


def build_schema_text() -> str:
    """Excel se asli column names aur 3 sample rows nikal ke prompt mein do."""
    cols = ", ".join(f"{c} ({df_data[c].dtype})" for c in df_data.columns)
    sample = df_data.head(3).to_csv(index=False)
    return f"Table name: {TABLE_NAME}\nColumns: {cols}\n\nSample rows (CSV):\n{sample}"


SYSTEM_PROMPT = f"""You are an expert data analyst. Convert the user's question
about airline ticket data into ONE SQLite SQL query.

{build_schema_text()}

Column meanings:
{COLUMN_HINTS}

Rules:
1. Use only the table "{TABLE_NAME}" and the exact column names listed above.
2. This is SQLite: use LIMIT (never TOP). Use SUM for totals, AVG for averages, COUNT for counts.
3. Only write a SELECT (or WITH ... SELECT) query. Never modify data.
4. Use NULLIF(x, 0) when dividing to avoid division by zero.
5. Give result columns clear aliases.
6. Reply with ONLY the SQL query, no explanation and no markdown.
7. If the question cannot be answered from this data, reply exactly: UNKNOWN
"""

EXAMPLES = [
    ("What is the total revenue generated from ticket sales?",
     f"SELECT SUM(Total_Actual_Payment_Amount) AS Total_Revenue FROM {TABLE_NAME};"),
    ("Which airline has the highest number of passengers?",
     f"SELECT Dominant_Operating_Airline, SUM(Total_Pax) AS Total_Passengers FROM {TABLE_NAME} "
     "GROUP BY Dominant_Operating_Airline ORDER BY Total_Passengers DESC LIMIT 1;"),
    ("Which origin city has the highest number of trips?",
     f"SELECT Trip_Origin_City, COUNT(*) AS Number_Of_Trips FROM {TABLE_NAME} "
     "GROUP BY Trip_Origin_City ORDER BY Number_Of_Trips DESC LIMIT 1;"),
]


def generate_sql_query(question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in EXAMPLES:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
        model=AI_MODEL, messages=messages, temperature=0
    )
    sql = (completion.choices[0].message.content or "").strip()
    # Agar AI ne ```sql ... ``` mein lapet diya ho toh hata do
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    return sql


def is_safe_select(sql: str) -> bool:
    """Sirf SELECT/WITH queries allow karo, ek hi statement."""
    body = sql.strip().rstrip(";").strip()
    if ";" in body:
        return False
    return bool(re.match(r"^(select|with)\b", body, flags=re.IGNORECASE))


def execute_sql_query(sql: str) -> pd.DataFrame:
    with db_lock:
        return pd.read_sql_query(sql, db)


def describe_answer(question: str, result: pd.DataFrame) -> str:
    """Result ko ek seedhe-saade English sentence mein samjhao."""
    preview = result.head(20).to_csv(index=False)
    completion = client.chat.completions.create(
        model=AI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "You answer questions about airline data. Using ONLY the query "
                "result given, reply in 1-3 short, clear sentences. Expand airline "
                "and airport codes to full names where you know them."
            )},
            {"role": "user", "content": f"Question: {question}\n\nQuery result:\n{preview}"},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# 4. Flask API
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "rows_loaded": len(df_data),
        "columns": list(df_data.columns),
    })


@app.route("/bot", methods=["POST"])
def bot():
    data = request.get_json(silent=True) or {}
    question = (data.get("Question") or "").strip()
    if not question:
        return jsonify({"error": "Request mein 'Question' bhejna zaroori hai."}), 400

    print(f"\nSawaal: {question}")

    try:
        sql = generate_sql_query(question)
    except Exception as e:
        print("Gemini error:", e)
        return jsonify({"error": f"Gemini AI se baat nahi ho payi: {e}"}), 502

    print(f"SQL: {sql}")

    if sql.upper().startswith("UNKNOWN"):
        return jsonify({
            "question": question, "sql": None, "data": [],
            "answer": "I do not know the answer to your question.",
        })

    if not is_safe_select(sql):
        return jsonify({"error": "Generated query safe nahi thi.", "sql": sql}), 400

    try:
        result = execute_sql_query(sql)
    except Exception as e:
        print("SQL error:", e)
        return jsonify({"error": f"Query chalane mein dikkat: {e}", "sql": sql}), 400

    # NaN / numpy types ko JSON-safe banane ka aasaan tarika
    rows = json.loads(result.to_json(orient="records"))

    if result.empty:
        answer = "No matching data was found for your question."
    else:
        try:
            answer = describe_answer(question, result)
        except Exception as e:
            print("Description error:", e)
            answer = json.dumps(rows[0])

    print(f"Jawab: {answer}")
    return jsonify({"question": question, "sql": sql, "data": rows, "answer": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5000)