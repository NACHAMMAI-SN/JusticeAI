import os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API"))
index = pc.Index("lawpal")

print("Loading embedding model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

legal_texts = [
    "Under the Hindu Marriage Act 1955, a wife has the right to claim maintenance and alimony after divorce.",
    "In India, child custody is decided based on the best interests of the child under the Guardianship Act.",
    "A woman can file for divorce on grounds of cruelty, adultery, or desertion under Section 13 of Hindu Marriage Act.",
    "Domestic violence cases are handled under the Protection of Women from Domestic Violence Act 2005.",
    "Property rights of daughters are equal to sons under the Hindu Succession Amendment Act 2005.",
    "An employer must pay salary within 7 days of end of wage period under Payment of Wages Act.",
    "A consumer can file complaint in District Consumer Forum for defective products under Consumer Protection Act 2019.",
    "Cheque bounce cases are filed under Section 138 of Negotiable Instruments Act.",
    "FIR can be filed at any police station under Section 154 of CrPC for cognizable offences.",
    "Bail application can be filed under Section 437 or 439 of CrPC depending on the offence.",
    "RTI application must be responded within 30 days under Right to Information Act 2005.",
    "Legal aid is available free of cost under Legal Services Authorities Act 1987 for eligible persons.",
    "Property disputes can be resolved through civil court or lok adalat for faster resolution.",
    "A will can be challenged in court on grounds of fraud, coercion or unsound mind of testator.",
    "Power of attorney can be given to trusted person to act on your behalf in legal matters.",
    "RERA complaint can be filed against builder for delay in possession under Real Estate Act 2016.",
    "Cybercrime complaints can be filed at cybercrime.gov.in or nearest police cyber cell.",
    "Employment termination must follow due process under Industrial Disputes Act 1947.",
    "GST fraud cases are handled by GST Intelligence and can result in criminal prosecution.",
    "Insurance claim disputes can be filed with Insurance Ombudsman for faster resolution.",
]

print("Creating embeddings...")
vectors = []
for i, text in enumerate(legal_texts):
    embedding = model.encode(text).tolist()
    vectors.append({
        "id": f"legal_{i}",
        "values": embedding,
        "metadata": {"text": text}
    })

print("Uploading to Pinecone...")
index.upsert(vectors=vectors)
print(f"✅ Successfully uploaded {len(vectors)} legal documents to Pinecone!")