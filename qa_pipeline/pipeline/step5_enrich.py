import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any


def classify_question_type(question: str) -> str:
    """Classify question into academic-standard types.
    
    Categories follow the standard QA taxonomy used in legal NLP research:
    - factoid: Direct fact lookup (who, what, which article)
    - yes_no: Binary answer questions
    - comparison: Compare two or more legal provisions
    - conditional: If-then scenarios
    - definition: "What is X" type questions
    - procedural: Process/procedure questions
    - multi_hop: Questions requiring reasoning across multiple sources
    """
    q = question.lower().strip()
    
    # Yes/No patterns (must check first — most specific)
    yes_no_patterns = [
        r"có\s+(được|phải|cần|đúng|bắt buộc)",
        r"(được|có thể)\s+.*\s+không\s*\?",
        r"có\s+.*\s+hay\s+không",
        r"đúng không",
        r"phải không",
    ]
    for pat in yes_no_patterns:
        if re.search(pat, q):
            return "yes_no"
    
    # Conditional patterns
    conditional_patterns = [
        r"nếu\s+.*\s+thì",
        r"trường\s+hợp\s+.*\s+(thì|sẽ|phải|được)",
        r"khi\s+nào\s+.*\s+(được|phải|cần)",
        r"trong\s+trường\s+hợp",
    ]
    for pat in conditional_patterns:
        if re.search(pat, q):
            return "conditional"
    
    # Comparison patterns
    comparison_patterns = [
        r"(so\s+sánh|khác\s+nhau|giống\s+nhau|sự\s+khác\s+biệt)",
        r"(khác\s+gì|giống\s+gì)",
        r"(A\s+và\s+B|giữa\s+.*\s+và\s+)",
    ]
    for pat in comparison_patterns:
        if re.search(pat, q):
            return "comparison"
    
    # Definition patterns
    definition_patterns = [
        r"(là\s+gì|được\s+hiểu\s+như\s+thế\s+nào|nghĩa\s+là\s+gì)",
        r"(định\s+nghĩa|khái\s+niệm|nội\s+dung)",
    ]
    for pat in definition_patterns:
        if re.search(pat, q):
            return "definition"
    
    # Procedural patterns
    procedural_patterns = [
        r"(thủ\s+tục|quy\s+trình|trình\s+tự|các\s+bước|hồ\s+sơ)",
        r"(cách\s+thức|làm\s+thế\s+nào|như\s+thế\s+nào)",
    ]
    for pat in procedural_patterns:
        if re.search(pat, q):
            return "procedural"
    
    # Default: factoid (direct lookup)
    return "factoid"


def assign_routing_label(sample: Dict[str, Any]) -> Dict[str, Any]:
    articles = sample.get("relevant_articles", [])
    # Default to 1 if empty or not present to be safe, though parse step enforces 1
    hop_count = len(articles) if articles else 1
    
    law_ids = set(a.get("law_id") for a in articles if "law_id" in a)
    is_cross_doc = len(law_ids) > 1

    if hop_count <= 1 and not is_cross_doc:
        routing_label = "dense_retrieval"
    elif hop_count >= 2 and not is_cross_doc:
        routing_label = "graph_traversal"
    else:
        routing_label = "hybrid_reasoning"

    # Difficulty score
    question = sample.get("question", "")
    reasoning_keywords = ["nếu", "trường hợp", "khi nào", "có được", 
                          "có phải", "được không", "như thế nào"]
    has_reasoning = any(kw in question.lower() for kw in reasoning_keywords)

    difficulty = round(
        0.4 * min(hop_count / 3, 1.0) +
        0.3 * int(is_cross_doc) +
        0.3 * int(has_reasoning),
        2
    )

    # Classify question type
    question_type = classify_question_type(question)
    
    # Track provenance (augmented or original)
    augmented = sample.get("augmented", False)

    return {
        **sample,
        "hop_count": hop_count,
        "is_cross_doc": is_cross_doc,
        "routing_label": routing_label,
        "difficulty": difficulty,
        "has_reasoning": has_reasoning,
        "question_type": question_type,
        "augmented": augmented,
    }

def step5_enrich(input_path: str | Path, output_checkpoint: str | Path) -> None:
    """
    Step 5: Enrich data with routing_label, hop_count, is_cross_doc, and difficulty.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found. Please run Step 4 first.")
        return

    print(f"✅ Loading Step 4 checkpoint: {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    enriched_data = []
    
    routing_counts = Counter()
    difficulties = []
    reasoning_count = 0
    
    for sample in data:
        enriched = assign_routing_label(sample)
        enriched_data.append(enriched)

        routing_counts[enriched["routing_label"]] += 1
        difficulties.append(enriched["difficulty"])
        if enriched.get("has_reasoning", False):
            reasoning_count += 1
            
    # Calculate difficulty stats
    if difficulties:
        min_diff = min(difficulties)
        max_diff = max(difficulties)
        mean_diff = sum(difficulties) / len(difficulties)
    else:
        min_diff = max_diff = mean_diff = 0.0
        
    buckets = {"0.0-0.3": 0, "0.3-0.6": 0, "0.6-1.0": 0}
    for d in difficulties:
        if d <= 0.3:
            buckets["0.0-0.3"] += 1
        elif d <= 0.6:
            buckets["0.3-0.6"] += 1
        else:
            buckets["0.6-1.0"] += 1
            
    print("\n✅ Step 5 Enrichment Complete!")
    print("\n1. Routing Label Counter:")
    for label, count in routing_counts.items():
        print(f"   - {label}: {count}")
        
    print("\n2. Difficulty Score Distribution:")
    print(f"   - Min : {min_diff:.2f}")
    print(f"   - Max : {max_diff:.2f}")
    print(f"   - Mean: {mean_diff:.2f}")
    print("   - Buckets:")
    for b_name, b_count in buckets.items():
        print(f"     * {b_name}: {b_count}")
        
    print(f"\n3. Samples with has_reasoning=True: {reasoning_count}")
    
    # Question type distribution
    qtype_counts = Counter(s.get("question_type", "unknown") for s in enriched_data)
    print("\n4. Question Type Distribution:")
    for qt, count in qtype_counts.most_common():
        print(f"   - {qt}: {count}")
    
    print("\n4. Example sample:")
    # Remove large answer/evidence strings from print to make it readable
    example = enriched_data[0].copy()
    example["answer"] = example["answer"][:50] + "..." if len(example.get("answer", "")) > 50 else example.get("answer", "")
    example["evidence"] = example["evidence"][:50] + "..." if len(example.get("evidence", "")) > 50 else example.get("evidence", "")
    print(json.dumps(example, ensure_ascii=False, indent=2))
    
    out_file = Path(output_checkpoint)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(enriched_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Checkpoint saved to: {out_file}")

if __name__ == "__main__":
    INPUT_PATH = "qa_pipeline/data/checkpoints/step4_deduped.json"
    OUTPUT_CHECKPOINT = "qa_pipeline/data/checkpoints/step5_enriched.json"
    step5_enrich(INPUT_PATH, OUTPUT_CHECKPOINT)
