import sys
import os
from pathlib import Path

# Thêm root vào sys.path để import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from router.two_stage_router import TwoStageRouter
from pipeline.conversation_manager import ConversationManager
import yaml
from loguru import logger

# Tắt log để output đẹp hơn
logger.remove()

def main():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    router = TwoStageRouter(config)
    
    queries = [
        ("Chitchat", "Xin chào trợ lý, bạn có thể giúp tôi được không?"),
        ("Dense Retrieval", "Mức xử phạt đối với hành vi đi xe máy không đội mũ bảo hiểm theo Nghị định 100/2019/NĐ-CP là bao nhiêu tiền?"),
        ("Graph Traversal", "Công ty vi phạm Điểm a Khoản 1 Điều 10 Nghị định 15 thì bị tước giấy phép bao lâu?"),
        ("Hybrid Reasoning", "Theo Luật Doanh nghiệp và Nghị định 01/2021/NĐ-CP, đồng thời tham chiếu thông tư 01/2021/TT-BKHĐT, thủ tục đăng ký thành lập công ty TNHH có điểm gì cần chú ý?"),
        ("Clarify", "Vậy mức phạt của hành vi đó là bao nhiêu?")
    ]
    
    print("="*60)
    print("DEMO PHÂN TÍCH QUERRY TỪNG BƯỚC CỦA TWO-STAGE ROUTER")
    print("="*60)
    
    for route_name, query in queries:
        print(f"\n[{route_name.upper()}]")
        print(f"Câu hỏi: \"{query}\"")
        
        # Chạy router (không có lịch sử)
        output = router.route(query=query, history="")
        
        print(f"-> Luồng quyết định: {output.route} (Độ tự tin: {output.confidence:.2f})")
        print(f"-> Lý do (Reasoning): {output.reasoning}")
        
        features = output.features
        print("-> Phân tích đặc trưng nổi bật:")
        
        if route_name == "Chitchat":
            print("   (Bắt được bằng Regex Heuristic, không cần bóc tách đặc trưng NLP)")
        else:
            # In ra các feature quan trọng
            fd = features._dict
            print(f"   - Mức độ phức tạp (Complexity): {fd.get('complexity_level', 0)}")
            print(f"   - Số câu hỏi phụ (Sub-questions): {fd.get('sub_question_count', 0)}")
            print(f"   - Tính chất Factoid (hỏi đáp nhanh): {'Có' if fd.get('is_factoid_question', 0) > 0 else 'Không'}")
            print(f"   - Số lượng thực thể (Entities): {fd.get('entity_count', 0)}")
            print(f"   - Độ sâu câu điều kiện (Nếu-Thì): {fd.get('conditional_depth', 0)}")
            
            ctx = features._ctx
            if ctx.get("has_pronoun") or ctx.get("missing_entity"):
                print(f"   - Phát hiện mơ hồ/thiếu thông tin: Có (Đại từ: {ctx.get('has_pronoun')} | Thiếu thực thể: {ctx.get('missing_entity')})")

if __name__ == "__main__":
    main()
