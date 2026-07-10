import yaml
import logging
from pprint import pprint
from router.two_stage_router import TwoStageRouter

# Tắt log thừa để dễ nhìn
logging.getLogger("router").setLevel(logging.ERROR)

def main():
    print("Khởi tạo Router...")
    with open('configs/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    router = TwoStageRouter(config)

    # Các trường hợp test đại diện cho từng loại câu hỏi và các lỗi đã sửa
    test_cases = [
        {
            "desc": "[Chitchat] Câu chào hỏi bình thường",
            "query": "hi, xin chào bạn",
            "expected_route": "chitchat"
        },
        {
            "desc": "[Lỗi 1 & 2 Fixed] Câu factoid có chứa số hiệu văn bản (Nghị định 15) và từ khóa 'bao lâu'",
            "query": "Công ty vi phạm Điểm a Khoản 1 Điều 10 Nghị định 15 thì bị tước giấy phép bao lâu?",
            "expected_route": "dense_retrieval"
        },
        {
            "desc": "[Lỗi 3, 4, 5 & 6 Fixed] Câu có cấu trúc điều kiện (Nếu...thì) nhưng KHÔNG chứa số hiệu văn bản (Out-of-distribution)",
            "query": "Tôi là Việt kiều Mỹ muốn mua nhà tại Việt Nam thì có được cấp sổ đỏ không? Nếu tôi mang thế chấp cho ngân hàng nước ngoài thì có hợp pháp không?",
            "expected_route": "hybrid_reasoning"
        },
        {
            "desc": "[Graph Traversal] Truy vấn đa bước nội bộ 1 văn bản (liên kết nhiều Điều)",
            "query": "Trình tự thu hồi đất vì mục đích quốc phòng theo Điều 61 và Điều 62 Luật Đất đai 2024 khác nhau như thế nào?",
            "expected_route": "graph_traversal"
        },
        {
            "desc": "[Clarify] Câu hỏi thiếu chủ thể hoặc tối nghĩa",
            "query": "Mức phạt là bao nhiêu?",
            "expected_route": "clarify"
        }
    ]

    print("=" * 80)
    print("BẮT ĐẦU DEMO ĐỊNH TUYẾN (ROUTING)\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}] {case['desc']}")
        print(f"Query: '{case['query']}'")
        
        # Route query
        result = router.route(case['query'])
        
        # In kết quả
        print(f"-> Giai đoạn 1 (XGBoost): {result.stage1_route} (Độ tự tin: {result.stage1_confidence:.3f})")
        if result.stage2_invoked:
            print(f"-> Đã KÍCH HOẠT Giai đoạn 2 (LLM Verifier) do: {result.stage2_trigger_reasons}")
            if result.stage2_override:
                print(f"-> Giai đoạn 2 đã GHI ĐÈ kết quả thành: {result.route}")
        
        print(f"==> KẾT QUẢ ĐỊNH TUYẾN CUỐI CÙNG: {result.route} (Kỳ vọng: {case['expected_route']})")
        
        # In một số đặc trưng quan trọng
        feats = result.features
        if feats and hasattr(feats, 'feature_dict'):
            print("Một số đặc trưng quan trọng:")
            print(f"  - legal_reference_count: {feats.legal_reference_count}")
            print(f"  - is_factoid: {feats.is_factoid}")
            print(f"  - conditional_depth: {feats.conditional_depth}")
            print(f"  - graph_keyword_count: {feats.graph_keyword_count}")
        print("-" * 80)

if __name__ == "__main__":
    main()
