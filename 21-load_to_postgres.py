import json
import psycopg2

conn = psycopg2.connect(host="localhost", port=5433, user="student",
                        password="othmen123", dbname="invoices")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS invoices (
    id SERIAL PRIMARY KEY,
    source_file TEXT UNIQUE,
    invoice_number TEXT,
    date DATE,
    total NUMERIC,
    shipping NUMERIC,
    discount_percent NUMERIC,
    discount_amount NUMERIC,
    bill_to TEXT,
    ship_to TEXT
)
""")

with open("metadata/invoices.json", encoding="utf-8") as f:
    invoices = json.load(f)

for inv in invoices:
    cur.execute("""
        INSERT INTO invoices
        (source_file, invoice_number, date, total, shipping,
         discount_percent, discount_amount, bill_to, ship_to)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (source_file) DO NOTHING
    """, (
        inv.get("source_file"), inv.get("invoice_number"), inv.get("date"),
        inv.get("total"), inv.get("shipping"),
        inv.get("discount_percent"), inv.get("discount_amount"),
        inv.get("bill_to"), inv.get("ship_to"),
    ))

conn.commit()
cur.close()
conn.close()
print(f"Inserted {len(invoices)} invoices.")