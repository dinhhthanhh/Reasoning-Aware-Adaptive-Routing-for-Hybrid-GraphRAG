import json
from pathlib import Path
from collections import Counter
from datetime import datetime

def generate_report(root_dir: Path) -> None:
    reports_dir = root_dir / "qa_pipeline/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "final_report.md"

    # Paths to all checkpoints
    checkpoints_dir = root_dir / "qa_pipeline/data/checkpoints"
    final_dir = root_dir / "qa_pipeline/data/final"

    step2_file = checkpoints_dir / "step2_parsed.json"
    step3_file = checkpoints_dir / "step3_filtered.json"
    step6b_file = checkpoints_dir / "step6b_augmented.json"
    
    train_file = final_dir / "train.json"
    dev_file = final_dir / "dev.json"
    test_file = final_dir / "test.json"

    def count_samples(filepath: Path) -> int:
        if filepath.exists():
            with filepath.open("r", encoding="utf-8") as f:
                return len(json.load(f))
        return 0

    # 1. Pipeline Execution Tracking
    initial_count = count_samples(step2_file)
    clean_count = count_samples(step3_file)
    augmented_count = count_samples(step6b_file)

    truncated_removed = initial_count - clean_count

    # 2. Final Data Distribution Tracking
    train_data, dev_data, test_data = [], [], []
    if train_file.exists():
        with train_file.open("r", encoding="utf-8") as f:
            train_data = json.load(f)
    if dev_file.exists():
        with dev_file.open("r", encoding="utf-8") as f:
            dev_data = json.load(f)
    if test_file.exists():
        with test_file.open("r", encoding="utf-8") as f:
            test_data = json.load(f)

    all_final_data = train_data + dev_data + test_data
    total_final = len(all_final_data)

    routing_counts = Counter(s.get("routing_label", "unknown") for s in all_final_data)
    hop_counts = Counter(s.get("hop_count", 0) for s in all_final_data)
    cross_doc_count = sum(1 for s in all_final_data if s.get("is_cross_doc"))

    # Difficulty Tracking
    difficulties = [s.get("difficulty", 0.0) for s in all_final_data]
    mean_diff = sum(difficulties) / len(difficulties) if difficulties else 0.0
    max_diff = max(difficulties) if difficulties else 0.0

    # Build the Markdown Report
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Báo Cáo Xây Dựng Dataset QA Hệ Thống Legal GraphRAG
**Ngày tạo:** {now_str}
**Thực hiện:** Đội ngũ Kỹ sư AI

---

## 1. Tóm tắt Tiết trình Xử lý Dữ liệu (Data Pipeline)

Hệ thống Data Pipeline đã được thiết kế nghiêm ngặt đi qua 7 bước chuẩn hóa, làm sạch, và nâng cấp chất lượng bằng mô hình LLM. Dưới đây là kết quả qua các giai đoạn:

| Giai đoạn | Số lượng (QA Pairs) | Ghi chú |
| --- | --- | --- |
| Dữ liệu thô (sau Parse) | **{initial_count}** | Dữ liệu đầu vào từ quá trình thu thập ban đầu |
| Lọc câu bị cắt cụt (Step 3) | **{clean_count}** | Loại bỏ {truncated_removed} câu trả lời không đầy đủ (Truncated) |
| Sau khi Augment (Step 6b) | **{augmented_count}** | Sinh thêm {augmented_count - clean_count} mẫu đa hop (Track B & C) nhờ Qwen LLM |

---

## 2. Phân bố Tập dữ liệu Cuối cùng (Final Dataset)

Bộ dữ liệu cuối cùng đảm bảo tính cân bằng và bao quát cho mô hình **Reasoning-Aware Adaptive Routing** có thể nhận biết được ý định câu hỏi để định tuyến tới chiến lược truy xuất (Retrieval Strategy) phù hợp.

### A. Phân bố Nhãn Định tuyến (Routing Labels)
Mô hình sẽ học cách phân loại các câu hỏi vào 3 chiến lược RAG chính:

- **Track A (dense_retrieval):** {routing_counts.get("dense_retrieval", 0)} mẫu. Chuyên xử lý các câu hỏi tra cứu thông tin trực diện (Single-hop).
- **Track B (graph_traversal):** {routing_counts.get("graph_traversal", 0)} mẫu. Chuyên xử lý các câu hỏi phức tạp yêu cầu truy vấn nhiều điều khoản trong *cùng 1 văn bản*.
- **Track C (hybrid_reasoning):** {routing_counts.get("hybrid_reasoning", 0)} mẫu. Chuyên xử lý các câu hỏi phân tích luật lệ chồng chéo, yêu cầu tổng hợp thông tin *từ 2 văn bản bộ/ngành khác nhau* (Cross-doc).

### B. Phân bố Hop-count và Cross-doc
- Câu hỏi Single-hop (1 Article): {hop_counts.get(1, 0)} mẫu.
- Câu hỏi Multi-hop (>= 2 Articles): {total_final - hop_counts.get(1, 0)} mẫu.
- Số lượng câu hỏi Cross-document: {cross_doc_count} mẫu.

### C. Độ khó của Câu hỏi (Difficulty Distribution)
Dữ liệu đã được gán nhãn độ khó dựa vào thuật toán heuristic kết hợp từ (từ khóa phân tích/suy luận, cross_doc flag, và hop_count):
- **Difficulty Mean (Trung bình):** {mean_diff:.2f} / 1.00
- **Difficulty Max (Cao nhất):** {max_diff:.2f} / 1.00

---

## 3. Quá trình chia dữ liệu cho Mô hình (Train/Dev/Test)

Dữ liệu được chia cực kỳ nghiêm ngặt bằng thuật toán Lấy mẫu phân tầng (Stratified Sampling) theo biến `routing_label` để đảm bảo tỷ lệ cân bằng ở cả 3 tệp nhỏ:
**Quy mô tổng quan:** {total_final} mẫu. Cấu trúc chia: 70/15/15.

| Loại tập | Số mẫu | Tỷ lệ (%) | Phân phối Nhãn (Track A / Track B / Track C) |
| --- | --- | --- | --- |
| **Train** | {len(train_data)} | {len(train_data)/total_final * 100:.1f}% | {Counter(s.get("routing_label") for s in train_data).most_common()} |
| **Dev** | {len(dev_data)} | {len(dev_data)/total_final * 100:.1f}% | {Counter(s.get("routing_label") for s in dev_data).most_common()} |
| **Test** | {len(test_data)} | {len(test_data)/total_final * 100:.1f}% | {Counter(s.get("routing_label") for s in test_data).most_common()} |

---

## 4. Kết luận chuẩn bị cho Huấn luyện
Toàn bộ quy trình sinh, làm sạch và tổng hợp dữ liệu cho GraphRAG Routing đã hoàn thành xuất sắc. 
Bộ dataset huấn luyện hiện đã được lưu tại thư mục `qa_pipeline/data/final/` và sẵn sàng cung cấp input tiêu chuẩn cho bài toán Phân loại Tuyến (Routing) thông qua kiến trúc AI sắp tới.
"""

    with report_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✅ Báo cáo tổng kết hoàn thành và được lưu tại: {report_path}")

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    generate_report(ROOT_DIR)
