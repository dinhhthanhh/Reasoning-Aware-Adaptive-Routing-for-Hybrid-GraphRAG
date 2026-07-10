import sys; sys.path.insert(0, ".")
from router.two_stage_router import TwoStageRouter

r = TwoStageRouter()
qs = [
    "Thuế giá trị gia tăng là gì?",
    "Chủ tịch Hội đồng có quyền điều hành cuộc họp theo quy định ở điều mấy?",
    "Cơ quan nào có thẩm quyền thành lập đơn vị trực thuộc Cục Hóa chất?",
    "Nếu chưa bố trí được Chủ tịch HĐND theo Nghị định 77/2026/NĐ-CP và Luật Giao dịch điện tử thì có được gia hạn không?",
    "Trường hợp đó thì xử lý thế nào?",   # pronoun/ambiguous
]

for q in qs:
    try:
        o = r.route(q)
        print(f"OK  {o.route:16} conf={o.confidence:.2f} stage2={o.stage2_invoked} | {q[:45]}")
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {e} | {q[:45]}")
