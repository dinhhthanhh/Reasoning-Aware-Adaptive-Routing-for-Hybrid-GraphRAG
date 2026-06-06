import json
import os
import time
import random
import yaml
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict, Counter

# Ensure we can import step5_enrich
sys.path.append(str(Path(__file__).parent))
try:
    from step5_enrich import assign_routing_label
except ImportError:
    print("❌ Error: Could not import assign_routing_label from step5_enrich.py")
    sys.exit(1)

import openai

# Use configs
def load_config():
    default_base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    default_model = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-32B-AWQ")
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    if not config_path.exists():
        return default_base_url, default_model
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        if config and "openai" in config:
            base_url = config["openai"].get("base_url", default_base_url)
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            model = config["openai"].get("model", default_model)
            return base_url, model
            
    return default_base_url, default_model

BASE_URL, MODEL = load_config()
API_KEY = os.environ.get("OPENAI_API_KEY", "not-required")

client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

PROMPT_B = """Bạn là chuyên gia pháp lý. Cho 2 điều luật sau từ cùng một văn bản:

[Điều A - {article_id_1}]: {text_1}
[Điều B - {article_id_2}]: {text_2}

Hãy sinh ra 2 câu hỏi mà để trả lời PHẢI kết hợp cả 2 điều trên.
Không được sinh câu hỏi chỉ cần 1 điều là đủ.

Trình bày kết quả dưới dạng JSON (mảng của các object chứa "question", "answer", "reasoning").
Tuyệt đối chỉ trả về dữ liệu JSON, không sử dụng markdown (không có ký hiệu ```json).
Mẫu JSON:
[
  {{
    "question": "...",
    "answer": "...",
    "reasoning": "Tại sao cần cả 2 điều?"
  }},
  {{
    "question": "...",
    "answer": "...",
    "reasoning": "Tại sao cần cả 2 điều?"
  }}
]
"""

PROMPT_C = """Bạn là chuyên gia pháp lý. Cho 2 điều luật từ 2 văn bản khác nhau:

[Văn bản 1 - {law_id_1}, {article_id_1}]: {text_1}
[Văn bản 2 - {law_id_2}, {article_id_2}]: {text_2}

Hãy sinh ra 1 câu hỏi phức tạp cần suy luận kết hợp cả 2 văn bản.
Câu hỏi nên dạng: "nếu... thì...", "trường hợp... có được... không?"

Trình bày kết quả dưới dạng JSON (mảng của các object chứa "question", "answer", "reasoning").
Tuyệt đối chỉ trả về dữ liệu JSON, không sử dụng markdown (không có ký hiệu ```json).
Mẫu JSON:
[
  {{
    "question": "...",
    "answer": "...",
    "reasoning": "Tại sao cần cả 2 văn bản?"
  }}
]
"""

def generate_qa_pairs(prompt: str) -> List[Dict[str, str]]:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048
        )
        content = response.choices[0].message.content.strip()
        
        # Parse JSON safely - strip formatting if added
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        content = content.strip()
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON Parse Error: {e}")
        print(f"Content was:\n{content}")
        return []
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return []

def quality_gate(sample: Dict[str, Any]) -> bool:
    """Basic quality gate: reject low-quality augmented samples."""
    q = sample.get("question", "").strip()
    a = sample.get("answer", "").strip()
    
    # Reject empty or very short
    if len(q) < 20 or len(a) < 30:
        return False
    # Reject if answer is just repeating the question
    if q.lower() in a.lower():
        return False
    # Reject if answer doesn't contain any Vietnamese characters
    if not any(c in a for c in 'àáảãạăắẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'):
        return False
    return True


def step6b_augment(input_path: str, output_checkpoint: str, corpus_path: str) -> None:
    input_file = Path(input_path)
    corpus_file = Path(corpus_path)
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found. Run Step 5 first.")
        return
    if not corpus_file.exists():
        print(f"❌ Error: Corpus {corpus_file} not found.")
        return

    print(f"✅ Loading Step 5 baseline data from: {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        base_data: List[Dict[str, Any]] = json.load(f)
        
    print(f"✅ Loading Full Corpus for context from: {corpus_file}")
    with corpus_file.open("r", encoding="utf-8") as f:
        articles = json.load(f)
        
    # Build dictionary of articles by doc_number
    doc_to_articles = defaultdict(list)
    for art in articles:
        doc_to_articles[art["doc_number"]].append(art)
        
    docs_with_multiple = [doc for doc, arts in doc_to_articles.items() if len(arts) >= 2]
    all_doc_numbers = list(doc_to_articles.keys())
    
    # Count existing base data labels to calculate augmentation targets
    base_counts = Counter(s.get("routing_label") for s in base_data)
    track_a = base_counts.get("dense_retrieval", 0)
    print(f"\n📊 Baseline distribution: {dict(base_counts)}")
    print(f"   Track A (dense): {track_a}")
    
    # Target: ~50/25/25 → Track B ≈ track_a/2, Track C ≈ track_a/2
    # With 710 Track A, we want ~355 Track B and ~355 Track C total
    # But we need to be pragmatic with API calls
    # Target: at least 250 Track B and 200 Track C total
    target_b = max(250, int(track_a * 0.35)) - base_counts.get("graph_traversal", 0)
    target_c = max(200, int(track_a * 0.30)) - base_counts.get("hybrid_reasoning", 0)
    
    calls_b = max(1, (target_b + 1) // 2)  # Each call produces ~2 QAs
    calls_c = max(1, target_c)  # Each call produces ~1 QA
    
    print(f"   Target new Track B: ~{target_b} (via {calls_b} API calls)")
    print(f"   Target new Track C: ~{target_c} (via {calls_c} API calls)")
    
    augmented_samples = []
    rejected = 0
    
    print(f"\n🚀 Starting Track B augmentation (Graph Traversal - Same Doc)")
    for i in range(calls_b):
        doc = random.choice(docs_with_multiple)
        arts = random.sample(doc_to_articles[doc], 2)
        
        prompt = PROMPT_B.format(
            article_id_1=arts[0]["article_id"],
            text_1=arts[0].get("content", ""),
            article_id_2=arts[1]["article_id"],
            text_2=arts[1].get("content", "")
        )
        
        print(f"   [Track B] Call {i+1}/{calls_b} (Doc: {doc})")
        qas = generate_qa_pairs(prompt)
        time.sleep(0.5)
        
        for qa in qas:
            gold_ctx = f"{arts[0].get('content', '')}\n\n{arts[1].get('content', '')}"
            new_sample = {
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "evidence": f"[Điều A]: {arts[0].get('content', '')}\n[Điều B]: {arts[1].get('content', '')}",
                "gold_context": gold_ctx,
                "article_key": f"{arts[0]['article_key']}, {arts[1]['article_key']}",
                "law": arts[0]["law"],
                "doc_number": doc,
                "url": arts[0]["url"],
                "relevant_articles": [
                    {"law_id": arts[0]["doc_number"], "article_id": arts[0]["article_id"],
                     "content": arts[0].get("content", "")},
                    {"law_id": arts[1]["doc_number"], "article_id": arts[1]["article_id"],
                     "content": arts[1].get("content", "")}
                ],
                "supporting_facts": [
                    {"law_id": arts[0]["doc_number"], "article_id": arts[0]["article_id"],
                     "title": f"{arts[0]['law']} — {arts[0]['article_id']}"},
                    {"law_id": arts[1]["doc_number"], "article_id": arts[1]["article_id"],
                     "title": f"{arts[1]['law']} — {arts[1]['article_id']}"}
                ],
                "augmented": True,
                "augmented_source": "track_b_same_doc",
            }
            # Quality gate
            if not quality_gate(new_sample):
                rejected += 1
                continue
            enriched = assign_routing_label(new_sample)
            augmented_samples.append(enriched)

    print(f"\n🚀 Starting Track C augmentation (Hybrid Reasoning - Cross Doc)")
    for i in range(calls_c):
        docs = random.sample(all_doc_numbers, 2)
        art1 = random.choice(doc_to_articles[docs[0]])
        art2 = random.choice(doc_to_articles[docs[1]])
        
        prompt = PROMPT_C.format(
            law_id_1=art1["doc_number"],
            article_id_1=art1["article_id"],
            text_1=art1.get("content", ""),
            law_id_2=art2["doc_number"],
            article_id_2=art2["article_id"],
            text_2=art2.get("content", "")
        )
        
        print(f"   [Track C] Call {i+1}/{calls_c} (Docs: {docs[0]} + {docs[1]})")
        qas = generate_qa_pairs(prompt)
        time.sleep(0.5)
        
        for qa in qas:
            gold_ctx = f"{art1.get('content', '')}\n\n{art2.get('content', '')}"
            new_sample = {
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "evidence": f"[Văn bản 1]: {art1.get('content', '')}\n[Văn bản 2]: {art2.get('content', '')}",
                "gold_context": gold_ctx,
                "article_key": f"{art1['article_key']}, {art2['article_key']}",
                "law": f"{art1['law']} & {art2['law']}",
                "doc_number": f"{art1['doc_number']}, {art2['doc_number']}",
                "url": f"{art1['url']}, {art2['url']}",
                "relevant_articles": [
                    {"law_id": art1["doc_number"], "article_id": art1["article_id"],
                     "content": art1.get("content", "")},
                    {"law_id": art2["doc_number"], "article_id": art2["article_id"],
                     "content": art2.get("content", "")}
                ],
                "supporting_facts": [
                    {"law_id": art1["doc_number"], "article_id": art1["article_id"],
                     "title": f"{art1['law']} — {art1['article_id']}"},
                    {"law_id": art2["doc_number"], "article_id": art2["article_id"],
                     "title": f"{art2['law']} — {art2['article_id']}"}
                ],
                "augmented": True,
                "augmented_source": "track_c_cross_doc",
            }
            if not quality_gate(new_sample):
                rejected += 1
                continue
            enriched = assign_routing_label(new_sample)
            augmented_samples.append(enriched)
            
    print(f"\n✅ Generated {len(augmented_samples)} new samples (rejected {rejected} by quality gate).")
    
    final_data = base_data + augmented_samples
    
    routing_counts = Counter(s.get("routing_label") for s in final_data)
    total = len(final_data)
    print(f"\n📊 Final Dataset Distribution ({total} total):")
    for label, count in routing_counts.most_common():
        pct = count / total * 100
        print(f"   - {label}: {count} ({pct:.1f}%)")

    out_file = Path(output_checkpoint)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Checkpoint saved to: {out_file}")

if __name__ == "__main__":
    # Configure variables based on the project paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    INPUT_PATH = ROOT_DIR / "qa_pipeline/data/checkpoints/step5_enriched.json"
    OUTPUT_CHECKPOINT = ROOT_DIR / "qa_pipeline/data/checkpoints/step6b_augmented.json"
    CORPUS_PATH = ROOT_DIR / "data/processed/articles.json"
    
    step6b_augment(str(INPUT_PATH), str(OUTPUT_CHECKPOINT), str(CORPUS_PATH))
