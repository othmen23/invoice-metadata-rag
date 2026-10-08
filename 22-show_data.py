import psycopg2

conn = psycopg2.connect(host="localhost", port=5433, user="student",
                        password="othmen123", dbname="invoices")
cur = conn.cursor()
cur.execute("""
    SELECT bill_to, date, total, shipping, discount_percent, ship_to
    FROM invoices
    ORDER BY date
""")
for row in cur.fetchall():
    print(row)

print("Total rows:", cur.rowcount)
conn.close()