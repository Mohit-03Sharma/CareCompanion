import sys
sys.path.insert(0, 'src')

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer
from retrieval import retrieve_dense, retrieve_hybrid, compute_confidence
from llm import generate_answer
from database import get_connection

model = SentenceTransformer("all-MiniLM-L6-v2")


# 20 health questions with ground truth answers
# Ground truth is a concise correct answer we use to score LLM responses
EVAL_DATASET = [
    {
        "query": "What are the symptoms of diabetes?",
        "ground_truth": "Symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurry vision, and slow healing sores."
    },
    {
        "query": "How is high blood pressure treated?",
        "ground_truth": "High blood pressure is treated with lifestyle changes like diet and exercise, and medications such as ACE inhibitors, beta blockers, and diuretics."
    },
    {
        "query": "What causes asthma?",
        "ground_truth": "Asthma is caused by airway inflammation and triggered by allergens, exercise, cold air, smoke, and respiratory infections."
    },
    {
        "query": "What are the risk factors for heart disease?",
        "ground_truth": "Risk factors for heart disease include high blood pressure, high cholesterol, smoking, obesity, diabetes, physical inactivity, and family history."
    },
    {
        "query": "How is depression diagnosed?",
        "ground_truth": "Depression is diagnosed through a clinical evaluation including questions about mood, sleep, appetite, energy, and duration of symptoms lasting at least two weeks."
    },
    {
        "query": "What is the treatment for anxiety?",
        "ground_truth": "Anxiety is treated with psychotherapy such as cognitive behavioral therapy, medications like SSRIs, and lifestyle changes including exercise and stress management."
    },
    {
        "query": "What are the early signs of Alzheimer's disease?",
        "ground_truth": "Early signs of Alzheimer's include memory loss, confusion, difficulty with problem solving, trouble completing familiar tasks, and changes in mood or personality."
    },
    {
        "query": "How can I lower my cholesterol?",
        "ground_truth": "Cholesterol can be lowered through a heart-healthy diet low in saturated fat, regular exercise, quitting smoking, and medications like statins."
    },
    {
        "query": "What are the symptoms of a stroke?",
        "ground_truth": "Stroke symptoms include sudden numbness or weakness in the face, arm or leg, confusion, trouble speaking, vision problems, and severe headache."
    },
    {
        "query": "What causes anemia?",
        "ground_truth": "Anemia is caused by iron deficiency, vitamin B12 deficiency, chronic disease, blood loss, or conditions that destroy red blood cells."
    },
    {
        "query": "How is asthma managed long term?",
        "ground_truth": "Long term asthma management includes daily controller medications, avoiding triggers, monitoring symptoms, and having a rescue inhaler available."
    },
    {
        "query": "What are the complications of obesity?",
        "ground_truth": "Obesity complications include heart disease, type 2 diabetes, high blood pressure, sleep apnea, joint problems, and certain cancers."
    },
    {
        "query": "What is the difference between type 1 and type 2 diabetes?",
        "ground_truth": "Type 1 diabetes is an autoimmune condition where the body produces no insulin. Type 2 diabetes is when the body does not use insulin effectively and is often linked to lifestyle factors."
    },
    {
        "query": "How is pneumonia treated?",
        "ground_truth": "Pneumonia is treated with antibiotics for bacterial pneumonia, rest, fluids, and fever reducers. Severe cases may require hospitalization."
    },
    {
        "query": "What are the symptoms of depression?",
        "ground_truth": "Depression symptoms include persistent sadness, loss of interest, fatigue, changes in sleep and appetite, difficulty concentrating, and feelings of worthlessness."
    },
    {
        "query": "What causes high blood pressure?",
        "ground_truth": "High blood pressure is caused by a combination of genetic factors, poor diet high in sodium, physical inactivity, obesity, stress, and aging."
    },
    {
        "query": "How is type 2 diabetes prevented?",
        "ground_truth": "Type 2 diabetes can be prevented through maintaining a healthy weight, eating a balanced diet, regular physical activity, and avoiding smoking."
    },
    {
        "query": "What are the warning signs of a heart attack?",
        "ground_truth": "Heart attack warning signs include chest pain or pressure, pain spreading to the arm or jaw, shortness of breath, sweating, nausea, and lightheadedness."
    },
    {
        "query": "How does sleep affect mental health?",
        "ground_truth": "Poor sleep is linked to increased risk of depression, anxiety, and mood disorders. Good sleep supports emotional regulation and cognitive function."
    },
    {
        "query": "What are the symptoms of COPD?",
        "ground_truth": "COPD symptoms include chronic cough, shortness of breath, wheezing, chest tightness, and frequent respiratory infections."
    },
]


def score_answer(answer, ground_truth):
    """
    Score an answer by computing cosine similarity against ground truth.
    Both are embedded and compared — higher score means more semantically similar.
    Scale is 0 to 1.
    """
    answer_embedding = model.encode(answer)
    truth_embedding = model.encode(ground_truth)

    # Cosine similarity between answer and ground truth
    similarity = np.dot(answer_embedding, truth_embedding) / (
        np.linalg.norm(answer_embedding) * np.linalg.norm(truth_embedding)
    )
    return float(similarity)


def run_experiment():
    """
    Run both strategies on all 20 questions.
    Score each answer, then compare statistically.
    """
    dense_scores = []
    hybrid_scores = []

    print(f"Running experiment on {len(EVAL_DATASET)} questions...\n")

    for i, item in enumerate(EVAL_DATASET):
        query = item["query"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/{len(EVAL_DATASET)}] {query[:60]}")

        # Strategy A — Dense
        try:
            dense_chunks = retrieve_dense(query)
            dense_answer, _ = generate_answer(query, dense_chunks)
            dense_score = score_answer(dense_answer, ground_truth)
        except Exception as e:
            print(f"  Dense error: {e}")
            dense_score = 0.0

        # Strategy B — Hybrid
        try:
            hybrid_chunks = retrieve_hybrid(query)
            hybrid_answer, _ = generate_answer(query, hybrid_chunks)
            hybrid_score = score_answer(hybrid_answer, ground_truth)
        except Exception as e:
            print(f"  Hybrid error: {e}")
            hybrid_score = 0.0

        dense_scores.append(dense_score)
        hybrid_scores.append(hybrid_score)
        print(f"  Dense: {dense_score:.3f} | Hybrid: {hybrid_score:.3f}")

        # Save to database
        save_experiment_result("dense_vs_hybrid", "dense", query, dense_score)
        save_experiment_result("dense_vs_hybrid", "hybrid", query, hybrid_score)

    return dense_scores, hybrid_scores


def save_experiment_result(experiment_name, strategy, query, relevance_score):
    """Save individual experiment result to database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO experiment_runs
            (experiment_name, strategy, query, relevance_score)
        VALUES (%s, %s, %s, %s)
    """, (experiment_name, strategy, query, relevance_score))
    conn.commit()
    cursor.close()
    conn.close()


def analyze_results(dense_scores, hybrid_scores):
    """
    Statistical comparison of both strategies.
    t-test: are the means significantly different?
    """
    dense_arr = np.array(dense_scores)
    hybrid_arr = np.array(hybrid_scores)

    # Paired t-test — same questions run through both strategies
    # Paired is correct here because each question is tested on both
    t_stat, p_value = stats.ttest_rel(hybrid_arr, dense_arr)

    # 95% confidence interval on the mean difference
    diff = hybrid_arr - dense_arr
    ci = stats.t.interval(
        0.95,
        len(diff) - 1,
        loc=np.mean(diff),
        scale=stats.sem(diff)
    )

    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"Questions evaluated:     {len(dense_scores)}")
    print(f"Dense mean score:        {np.mean(dense_arr):.4f}")
    print(f"Hybrid mean score:       {np.mean(hybrid_arr):.4f}")
    print(f"Mean difference:         {np.mean(diff):.4f}")
    print(f"t-statistic:             {t_stat:.4f}")
    print(f"p-value:                 {p_value:.4f}")
    print(f"95% CI on difference:    [{ci[0]:.4f}, {ci[1]:.4f}]")
    print()

    if p_value < 0.05:
        winner = "Hybrid" if np.mean(hybrid_arr) > np.mean(dense_arr) else "Dense"
        print(f"Result: {winner} is statistically significantly better (p < 0.05)")
    else:
        print("Result: No statistically significant difference between strategies (p >= 0.05)")

    print()
    print("Interpretation:")
    print(f"  The p-value of {p_value:.4f} means there is a {p_value*100:.1f}% probability")
    print(f"  of observing this difference if both strategies were equally good.")
    if p_value < 0.05:
        print("  This is below our 0.05 threshold — the difference is real, not random noise.")
    else:
        print("  This is above our 0.05 threshold — we cannot rule out random variation.")


if __name__ == "__main__":
    dense_scores, hybrid_scores = run_experiment()
    analyze_results(dense_scores, hybrid_scores)