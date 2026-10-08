import os
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

INVOICE_FOLDER = Path("data/invoices")
OUTPUT_FILE = Path("metadata/invoices.json")
MAX_FILES = 20  # test with 20 only

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
Extract metadata from this invoice text.
Return ONLY a JSON object with these keys:
- invoice_number (string, the number after #)
- date (string, format YYYY-MM-DD)
- total (number, no currency symbol)
- shipping (number, no currency symbol)
- discount_percent (number, e.g. 20)
- discount_amount (number, no currency symbol)
- bill_to (string, customer name)
- ship_to (string, full address on one line)

If a value is missing, use null. Do not add any explanation.

Invoice text:
{text}
""")

chain = prompt | llm | JsonOutputParser()

OUTPUT_FILE.parent.mkdir(exist_ok=True)
results = []

pdf_files = sorted(INVOICE_FOLDER.glob("*.pdf"))[:MAX_FILES]
print(f"Processing {len(pdf_files)} invoices...")

for pdf in pdf_files:
    try:
        pages = PyPDFLoader(str(pdf)).load()
        text = "\n".join(p.page_content for p in pages)
        data = chain.invoke({"text": text})
        data["source_file"] = pdf.name
        results.append(data)
        print("OK   ", pdf.name, "->", data["bill_to"], data["total"])
    except Exception as e:
        print("ERROR", pdf.name, e)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} invoices to {OUTPUT_FILE}")