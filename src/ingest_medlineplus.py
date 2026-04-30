import time
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from database import get_connection

model = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# MedlinePlus topic pages — static, reliable, no API key needed
# These are the most commonly searched health topics in the US
TOPICS = [
    ("Diabetes", "https://medlineplus.gov/diabetes.html"),
    ("Heart Disease", "https://medlineplus.gov/heartdisease.html"),
    ("High Blood Pressure", "https://medlineplus.gov/highbloodpressure.html"),
    ("Asthma", "https://medlineplus.gov/asthma.html"),
    ("Depression", "https://medlineplus.gov/depression.html"),
    ("Anxiety", "https://medlineplus.gov/anxiety.html"),
    ("Cancer", "https://medlineplus.gov/cancer.html"),
    ("Obesity", "https://medlineplus.gov/obesity.html"),
    ("Stroke", "https://medlineplus.gov/stroke.html"),
    ("Alzheimer's Disease", "https://medlineplus.gov/alzheimersdisease.html"),
    ("Arthritis", "https://medlineplus.gov/arthritis.html"),
    ("Asthma", "https://medlineplus.gov/asthma.html"),
    ("Back Pain", "https://medlineplus.gov/backpain.html"),
    ("Cholesterol", "https://medlineplus.gov/cholesterol.html"),
    ("COPD", "https://medlineplus.gov/copd.html"),
    ("COVID-19", "https://medlineplus.gov/covid19.html"),
    ("Flu", "https://medlineplus.gov/flu.html"),
    ("HIV/AIDS", "https://medlineplus.gov/hivaids.html"),
    ("Kidney Disease", "https://medlineplus.gov/kidneydiseases.html"),
    ("Liver Disease", "https://medlineplus.gov/liverdiseases.html"),
    ("Lung Cancer", "https://medlineplus.gov/lungcancer.html"),
    ("Mental Health", "https://medlineplus.gov/mentalhealth.html"),
    ("Migraine", "https://medlineplus.gov/migraine.html"),
    ("Nutrition", "https://medlineplus.gov/nutrition.html"),
    ("Osteoporosis", "https://medlineplus.gov/osteoporosis.html"),
    ("Pain", "https://medlineplus.gov/pain.html"),
    ("Parkinson's Disease", "https://medlineplus.gov/parkinsonsdisease.html"),
    ("Pneumonia", "https://medlineplus.gov/pneumonia.html"),
    ("Pregnancy", "https://medlineplus.gov/pregnancy.html"),
    ("Sleep Disorders", "https://medlineplus.gov/sleepdisorders.html"),
    ("Thyroid Diseases", "https://medlineplus.gov/thyroiddiseases.html"),
    ("Type 2 Diabetes", "https://medlineplus.gov/diabetestype2.html"),
    ("Vaccines", "https://medlineplus.gov/vaccines.html"),
    ("Weight Control", "https://medlineplus.gov/weightcontrol.html"),
    ("Alcohol", "https://medlineplus.gov/alcohol.html"),
    ("Allergies", "https://medlineplus.gov/allergy.html"),
    ("Anemia", "https://medlineplus.gov/anemia.html"),
    ("Autism", "https://medlineplus.gov/autismspectrumdisorder.html"),
    ("Blood Pressure", "https://medlineplus.gov/bloodpressure.html"),
    ("Breast Cancer", "https://medlineplus.gov/breastcancer.html"),
    ("Child Nutrition", "https://medlineplus.gov/childnutrition.html"),
    ("Colorectal Cancer", "https://medlineplus.gov/colorectalcancer.html"),
    ("Dementia", "https://medlineplus.gov/dementia.html"),
    ("Eczema", "https://medlineplus.gov/eczema.html"),
    ("Exercise", "https://medlineplus.gov/exerciseandphysicalfitness.html"),
    ("Food Safety", "https://medlineplus.gov/foodsafety.html"),
    ("Headache", "https://medlineplus.gov/headache.html"),
    ("Heart Attack", "https://medlineplus.gov/heartattack.html"),
    ("Infectious Diseases", "https://medlineplus.gov/infectiousdiseases.html"),
    ("Insomnia", "https://medlineplus.gov/insomnia.html"),
]


def scrape_topic_page(url):
    """Fetch a MedlinePlus page and return clean text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["nav", "header", "footer", "script", "style"]):
            tag.decompose()

        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            return None

        text = main.get_text(separator=" ", strip=True)
        return text if len(text) > 200 else None

    except Exception as e:
        print(f"  Error: {e}")
        return None


def chunk_text(text):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def save_chunks(topic_name, url, chunks, embeddings):
    """Save chunks and embeddings to database."""
    conn = get_connection()
    cursor = conn.cursor()
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        cursor.execute("""
            INSERT INTO knowledge_chunks
                (source, topic, url, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, ("medlineplus", topic_name, url, i, chunk, embedding.tolist()))
    conn.commit()
    cursor.close()
    conn.close()


def run_ingestion():
    """Main pipeline: scrape → chunk → embed → save."""
    total_chunks = 0

    for i, (name, url) in enumerate(TOPICS):
        print(f"[{i+1}/{len(TOPICS)}] {name}")

        text = scrape_topic_page(url)
        if not text:
            print(f"  Skipped — no content")
            continue

        chunks = chunk_text(text)
        embeddings = model.encode(chunks, show_progress_bar=False)
        save_chunks(name, url, chunks, embeddings)
        total_chunks += len(chunks)
        print(f"  {len(chunks)} chunks saved")

        time.sleep(0.5)

    print(f"\nDone. {total_chunks} total chunks saved.")


if __name__ == "__main__":
    run_ingestion()