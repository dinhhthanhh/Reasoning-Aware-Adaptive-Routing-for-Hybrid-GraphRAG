"""Build the internal conversation-aware ambiguity benchmark.

The output is a controlled stress test, not a public benchmark. It is designed
to evaluate whether the current router uses conversation history safely.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASE_SCENARIOS: list[dict[str, str]] = [
    {
        "query": "Văn bản đó còn hiệu lực không?",
        "route": "graph_traversal",
        "entity": "Nghị định 100/2019/NĐ-CP",
        "history": "Người dùng đang hỏi về Nghị định 100/2019/NĐ-CP về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ.",
        "irrelevant": "Người dùng hỏi cách tra cứu văn bản pháp luật trên cổng thông tin điện tử, nhưng chưa nêu văn bản cụ thể.",
        "conflict": "Người dùng nhắc đến Nghị định 100/2019/NĐ-CP và Luật Đường bộ 2024 trong cùng lượt hỏi.",
    },
    {
        "query": "Điều này quy định điều kiện gì?",
        "route": "dense_retrieval",
        "entity": "Điều 8 Luật Hôn nhân và gia đình",
        "history": "Trước đó người dùng hỏi về Điều 8 Luật Hôn nhân và gia đình liên quan đến điều kiện kết hôn.",
        "irrelevant": "Trước đó người dùng hỏi chung về khái niệm hôn nhân hợp pháp nhưng không nêu điều luật nào.",
        "conflict": "Trước đó cuộc hội thoại nhắc đến Điều 8 và Điều 10 của Luật Hôn nhân và gia đình.",
    },
    {
        "query": "Quy định đó áp dụng cho đối tượng nào?",
        "route": "dense_retrieval",
        "entity": "Điều 20 Luật Đường bộ",
        "history": "Người dùng vừa hỏi về Điều 20 Luật Đường bộ quy định yêu cầu khi xây dựng công trình hạ tầng kỹ thuật sử dụng chung với đường bộ.",
        "irrelevant": "Người dùng vừa hỏi về cách phân biệt đường bộ và đường sắt ở mức khái niệm.",
        "conflict": "Người dùng vừa nhắc đến Điều 20 Luật Đường bộ và Điều 73 Luật Đường bộ.",
    },
    {
        "query": "Thủ tục đó do cơ quan nào giải quyết?",
        "route": "graph_traversal",
        "entity": "thủ tục hành chính được phân cấp cho Cục Quản lý Y, Dược cổ truyền",
        "history": "Người dùng hỏi về thủ tục hành chính được phân cấp cho Cục Quản lý Y, Dược cổ truyền tại Quyết định của Bộ Y tế.",
        "irrelevant": "Người dùng hỏi chung về thủ tục hành chính nhưng không nêu lĩnh vực hoặc cơ quan.",
        "conflict": "Người dùng nhắc đến thủ tục mỹ phẩm của UBND tỉnh và thủ tục y dược cổ truyền của Bộ Y tế.",
    },
    {
        "query": "Nội dung trên có phải căn cứ để xử phạt không?",
        "route": "hybrid_reasoning",
        "entity": "quy định về quản lý chất thải rắn sinh hoạt tại tỉnh Thanh Hóa",
        "history": "Người dùng đang hỏi về quy định quản lý chất thải rắn sinh hoạt tại tỉnh Thanh Hóa và khả năng xử phạt vi phạm hành chính.",
        "irrelevant": "Người dùng hỏi về khái niệm xử phạt hành chính nói chung.",
        "conflict": "Người dùng nhắc đến quy định chất thải rắn sinh hoạt và quy định bảo vệ quyền lợi người tiêu dùng.",
    },
    {
        "query": "Văn bản đó sửa đổi quy định nào?",
        "route": "graph_traversal",
        "entity": "Quyết định 732/QĐ-UBND",
        "history": "Người dùng vừa hỏi về Quyết định 732/QĐ-UBND bãi bỏ một phần quy định quản lý chất thải rắn sinh hoạt.",
        "irrelevant": "Người dùng hỏi cách tìm văn bản sửa đổi nhưng chưa nêu số hiệu văn bản.",
        "conflict": "Người dùng nhắc đến Quyết định 732/QĐ-UBND và Quyết định 13/2022/QĐ-UBND.",
    },
    {
        "query": "Quy định này có được áp dụng đồng thời với quy định về ngân sách không?",
        "route": "hybrid_reasoning",
        "entity": "quy định chuyên ngành về khám bệnh, chữa bệnh",
        "history": "Người dùng đang hỏi một cơ quan y tế sử dụng ngân sách để thực hiện nhiệm vụ thuộc lĩnh vực khám bệnh, chữa bệnh.",
        "irrelevant": "Người dùng hỏi chung về nguyên tắc áp dụng ngân sách nhà nước.",
        "conflict": "Người dùng nhắc đến quy định ngân sách nhà nước, quy định khám bệnh chữa bệnh và quy định đầu tư công.",
    },
    {
        "query": "Điều khoản đó có hiệu lực từ khi nào?",
        "route": "graph_traversal",
        "entity": "Điều 22 Nghị định 77/2026/NĐ-CP",
        "history": "Người dùng vừa hỏi về Điều 22 Nghị định 77/2026/NĐ-CP quy định hiệu lực thi hành.",
        "irrelevant": "Người dùng hỏi chung về cách xác định thời điểm có hiệu lực của văn bản.",
        "conflict": "Người dùng nhắc đến Điều 19 và Điều 22 của Nghị định 77/2026/NĐ-CP.",
    },
    {
        "query": "Cơ quan đó có thẩm quyền ban hành quyết định không?",
        "route": "graph_traversal",
        "entity": "Ủy ban nhân dân cấp tỉnh",
        "history": "Người dùng đang hỏi về thẩm quyền của Ủy ban nhân dân cấp tỉnh trong việc ban hành quyết định quản lý địa phương.",
        "irrelevant": "Người dùng hỏi chung cơ quan nhà nước có thẩm quyền là gì.",
        "conflict": "Người dùng nhắc đến Bộ Y tế và Ủy ban nhân dân cấp tỉnh trong cùng bối cảnh.",
    },
    {
        "query": "Trường hợp này có được miễn giảm tiền thuê đất không?",
        "route": "hybrid_reasoning",
        "entity": "tổ chức bị thiệt hại tài sản do thiên tai khi xin miễn giảm tiền thuê đất",
        "history": "Người dùng hỏi về tổ chức bị thiệt hại tài sản do thiên tai và hồ sơ xin miễn giảm tiền thuê đất.",
        "irrelevant": "Người dùng hỏi chung về nghĩa vụ tài chính đất đai.",
        "conflict": "Người dùng nhắc đến người dân tộc thiểu số, người khuyết tật và thiệt hại do thiên tai trong cùng câu hỏi.",
    },
    {
        "query": "Văn bản trên hướng dẫn luật nào?",
        "route": "graph_traversal",
        "entity": "Nghị định 75/2026/NĐ-CP",
        "history": "Người dùng vừa hỏi về Nghị định 75/2026/NĐ-CP liên quan đến cơ chế tự chủ tài chính của cơ quan nhà nước.",
        "irrelevant": "Người dùng hỏi cách phân biệt nghị định và thông tư.",
        "conflict": "Người dùng nhắc đến Nghị định 75/2026/NĐ-CP và Nghị định 77/2026/NĐ-CP.",
    },
    {
        "query": "Quy định đó bị bãi bỏ bởi văn bản nào?",
        "route": "graph_traversal",
        "entity": "thủ tục hành chính trong lĩnh vực mỹ phẩm tại Vĩnh Long",
        "history": "Người dùng đang hỏi về thủ tục hành chính trong lĩnh vực mỹ phẩm bị bãi bỏ tại tỉnh Vĩnh Long.",
        "irrelevant": "Người dùng hỏi chung bãi bỏ thủ tục hành chính nghĩa là gì.",
        "conflict": "Người dùng nhắc đến Quyết định 1552/QĐ-UBND và Quyết định 846/QĐ-BNNMT.",
    },
    {
        "query": "Nội dung đó yêu cầu hồ sơ gồm những gì?",
        "route": "dense_retrieval",
        "entity": "hồ sơ đổi thẻ Sỹ quan kiểm tra tàu biển",
        "history": "Người dùng vừa hỏi về hồ sơ đổi thẻ Sỹ quan kiểm tra tàu biển trong văn bản hợp nhất của Bộ Xây dựng.",
        "irrelevant": "Người dùng hỏi chung về khái niệm hồ sơ hành chính.",
        "conflict": "Người dùng nhắc đến hồ sơ đổi thẻ Sỹ quan kiểm tra tàu biển và hồ sơ miễn giảm tiền thuê đất.",
    },
    {
        "query": "Thủ tục này có cần xin phép trước không?",
        "route": "dense_retrieval",
        "entity": "khám bệnh, chữa bệnh nhân đạo của tổ chức, cá nhân nước ngoài tại Việt Nam",
        "history": "Người dùng hỏi tổ chức, cá nhân nước ngoài muốn khám bệnh chữa bệnh nhân đạo tại Việt Nam cần được cơ quan nào cho phép.",
        "irrelevant": "Người dùng hỏi chung về thủ tục cấp phép trong lĩnh vực y tế.",
        "conflict": "Người dùng nhắc đến thủ tục khám bệnh nhân đạo và thủ tục điều chỉnh giấy phép hoạt động bệnh viện tư nhân.",
    },
    {
        "query": "Điều này liên quan đến trách nhiệm của ai?",
        "route": "graph_traversal",
        "entity": "Điều 12 Nghị định 75/2026/NĐ-CP",
        "history": "Người dùng đang hỏi về Điều 12 Nghị định 75/2026/NĐ-CP quy định trách nhiệm quản lý và sử dụng kinh phí hành chính.",
        "irrelevant": "Người dùng hỏi chung trách nhiệm pháp lý là gì.",
        "conflict": "Người dùng nhắc đến Điều 12 và Điều 13 của Nghị định 75/2026/NĐ-CP.",
    },
    {
        "query": "Văn bản đó có thay thế văn bản cũ không?",
        "route": "graph_traversal",
        "entity": "Thông tư 02/2026/TT-BYT",
        "history": "Người dùng hỏi về Thông tư 02/2026/TT-BYT và việc xử lý hồ sơ đã nộp trước ngày văn bản có hiệu lực.",
        "irrelevant": "Người dùng hỏi chung về nguyên tắc chuyển tiếp khi văn bản mới có hiệu lực.",
        "conflict": "Người dùng nhắc đến Thông tư 02/2026/TT-BYT và Thông tư 10/VBHN-BYT.",
    },
    {
        "query": "Quy định này áp dụng cho doanh nghiệp nào?",
        "route": "dense_retrieval",
        "entity": "quy định về lưu trữ hồ sơ nghiệp vụ bảo hiểm",
        "history": "Người dùng hỏi doanh nghiệp kinh doanh bảo hiểm có phải lưu trữ hồ sơ nghiệp vụ theo Luật Kinh doanh Bảo hiểm không.",
        "irrelevant": "Người dùng hỏi chung doanh nghiệp là gì trong pháp luật.",
        "conflict": "Người dùng nhắc đến doanh nghiệp bảo hiểm và doanh nghiệp vận tải hàng hóa bằng xe ô tô.",
    },
    {
        "query": "Trường hợp đó phải căn cứ những nhóm quy định nào?",
        "route": "hybrid_reasoning",
        "entity": "Sở Khoa học và Công nghệ thực hiện thủ tục xây dựng đồng thời tuân thủ kế hoạch chuyển đổi số",
        "history": "Người dùng hỏi Sở Khoa học và Công nghệ vừa thực hiện thủ tục xây dựng vừa phải tuân thủ kế hoạch chuyển đổi số.",
        "irrelevant": "Người dùng hỏi chung cách tìm căn cứ pháp lý cho cơ quan nhà nước.",
        "conflict": "Người dùng nhắc đến thủ tục xây dựng, thủ tục thuế và kế hoạch chuyển đổi số.",
    },
    {
        "query": "Điều khoản này có ngoại lệ nào không?",
        "route": "dense_retrieval",
        "entity": "điều khoản về điều kiện kết hôn",
        "history": "Người dùng đang hỏi điều khoản về điều kiện kết hôn trong Luật Hôn nhân và gia đình.",
        "irrelevant": "Người dùng hỏi chung ngoại lệ trong quy định pháp luật là gì.",
        "conflict": "Người dùng nhắc đến điều kiện kết hôn và trường hợp cấm kết hôn.",
    },
    {
        "query": "Văn bản đó được cơ quan nào ban hành?",
        "route": "graph_traversal",
        "entity": "Quyết định 846/QĐ-BNNMT",
        "history": "Người dùng vừa hỏi về Quyết định 846/QĐ-BNNMT công bố thủ tục hành chính trong lĩnh vực đê điều và phòng chống thiên tai.",
        "irrelevant": "Người dùng hỏi chung cơ quan ban hành văn bản là gì.",
        "conflict": "Người dùng nhắc đến Quyết định 846/QĐ-BNNMT và Quyết định 624/QĐ-BGDĐT.",
    },
]


MISSING_ENTITY_QUERIES = [
    "Mức phạt trong trường hợp này là bao nhiêu?",
    "Thủ tục này thực hiện như thế nào?",
    "Quy định mới nhất về vấn đề này là gì?",
    "Có được làm như vậy không?",
    "Cơ quan nào có thẩm quyền trong trường hợp đó?",
    "Hồ sơ cần chuẩn bị gồm những giấy tờ nào?",
    "Khi nào quy định này có hiệu lực?",
    "Ai phải chịu trách nhiệm về việc này?",
    "Trường hợp này có bị xử phạt không?",
    "Có cần xin phép trước khi thực hiện không?",
    "Quy định này áp dụng cho những đối tượng nào?",
    "Nội dung đó được hướng dẫn ở đâu?",
    "Văn bản liên quan là văn bản nào?",
    "Điều khoản nào cho phép thực hiện việc này?",
    "Có được tiếp tục xử lý theo quy định cũ không?",
    "Cần căn cứ vào luật nào để trả lời?",
    "Phải báo cáo cho cơ quan nào?",
    "Mức hỗ trợ được tính như thế nào?",
    "Có phải nộp thêm hồ sơ không?",
    "Thời hạn giải quyết là bao lâu?",
]


MULTI_INTERPRETATION_QUERIES = [
    "Người lao động có được nghỉ không?",
    "Doanh nghiệp có bị phạt không?",
    "Trường hợp này có phải xin phép không?",
    "Cơ quan nhà nước có được tự quyết định không?",
    "Người dân có được khiếu nại không?",
    "Tổ chức nước ngoài có được hoạt động tại Việt Nam không?",
    "Dự án có được tiếp tục triển khai không?",
    "Hồ sơ đã nộp có được xử lý tiếp không?",
    "Có được áp dụng quy định cũ không?",
    "Có được miễn giảm nghĩa vụ tài chính không?",
    "Cá nhân có được cấp giấy phép không?",
    "Công ty có phải lưu trữ hồ sơ không?",
    "Ủy ban nhân dân có thẩm quyền không?",
    "Bộ quản lý ngành có phải hướng dẫn không?",
    "Quy định mới có áp dụng ngay không?",
    "Có được sử dụng nguồn kinh phí này không?",
    "Có phải công bố thủ tục hành chính không?",
    "Có được đồng thời áp dụng hai quy định không?",
    "Có phải thực hiện báo cáo định kỳ không?",
    "Văn bản có hết hiệu lực không?",
]


DENSE_CONTROLS = [
    ("Điều kiện kết hôn theo Luật Hôn nhân và gia đình gồm những gì?", "điều kiện kết hôn"),
    ("Hành lang an toàn đường bộ là gì?", "hành lang an toàn đường bộ"),
    ("Hồ sơ đổi thẻ Sỹ quan kiểm tra tàu biển gồm những gì?", "hồ sơ đổi thẻ sỹ quan kiểm tra tàu biển"),
    ("Xe ô tô cứu thương vận tải người bệnh phải có những trang thiết bị nào?", "trang thiết bị xe cứu thương"),
    ("Nguồn thu từ dịch vụ sử dụng đường bộ được điều chỉnh theo quy định nào?", "nguồn thu dịch vụ sử dụng đường bộ"),
    ("Đất dành cho kết cấu hạ tầng đường bộ bao gồm những loại đất nào?", "đất kết cấu hạ tầng đường bộ"),
    ("Nội dung của một phép bay bao gồm những gì?", "nội dung phép bay"),
    ("Cá nhân có được sử dụng xe bốn bánh có gắn động cơ để chở người nội bộ không?", "xe bốn bánh chở người nội bộ"),
    ("Định mức sử dụng giấy A4 cho mỗi xã là bao nhiêu?", "định mức giấy A4"),
    ("Ai có trách nhiệm lập kế hoạch hoạt động bay dân dụng theo mùa và theo ngày?", "kế hoạch hoạt động bay dân dụng"),
    ("Tổ chức nước ngoài muốn khám bệnh nhân đạo tại Việt Nam cần được cơ quan nào cho phép?", "khám bệnh nhân đạo"),
    ("Cơ quan thực hiện chế độ tự chủ có trách nhiệm gì trong quản lý kinh phí hành chính?", "trách nhiệm quản lý kinh phí hành chính"),
    ("Việc khai thác tài sản kết cấu hạ tầng đường cao tốc được thực hiện như thế nào?", "khai thác tài sản hạ tầng đường cao tốc"),
    ("Bộ Quốc phòng có vai trò gì trong việc thiết lập khu vực bay và đường bay?", "vai trò Bộ Quốc phòng trong khu vực bay"),
    ("Vùng trời sân bay được xác định dựa trên những yếu tố nào?", "vùng trời sân bay"),
    ("Công ty vận chuyển chất thải rắn xây dựng cần lưu ý gì về phương tiện vận chuyển?", "phương tiện vận chuyển chất thải rắn"),
    ("Cơ quan nào chịu trách nhiệm chủ trì xác định khu vực xả nhiên liệu từ tàu bay dân dụng?", "khu vực xả nhiên liệu tàu bay"),
    ("Thông tư này quy định những nội dung nào liên quan đến sỹ quan kiểm tra tàu biển?", "sỹ quan kiểm tra tàu biển"),
    ("Tôi cần điều kiện gì để đổi thẻ Sỹ quan kiểm tra tàu biển khi hết hạn?", "đổi thẻ sỹ quan kiểm tra tàu biển"),
    ("Phạm vi đất để bảo vệ, bảo trì đường bộ có nền đắp được xác định như thế nào?", "đất bảo vệ bảo trì đường bộ"),
]


GRAPH_HYBRID_CONTROLS = [
    ("Quyết định 732/QĐ-UBND bãi bỏ hoặc sửa đổi quy định nào?", "graph_traversal", "Quyết định 732/QĐ-UBND"),
    ("Văn bản nào bãi bỏ thủ tục hành chính trong lĩnh vực mỹ phẩm tại Vĩnh Long?", "graph_traversal", "thủ tục mỹ phẩm Vĩnh Long"),
    ("Thông tư 02/2026/TT-BYT có quy định chuyển tiếp cho hồ sơ đã nộp trước ngày hiệu lực không?", "graph_traversal", "Thông tư 02/2026/TT-BYT"),
    ("Điều khoản nào quy định hiệu lực thi hành của Nghị định 77/2026/NĐ-CP?", "graph_traversal", "Nghị định 77/2026/NĐ-CP"),
    ("Quyết định 846/QĐ-BNNMT do cơ quan nào ban hành và công bố thủ tục nào?", "graph_traversal", "Quyết định 846/QĐ-BNNMT"),
    ("Các thủ tục hành chính nào bị bãi bỏ theo Quyết định 1552/QĐ-UBND và văn bản nào quy định việc bãi bỏ?", "graph_traversal", "Quyết định 1552/QĐ-UBND"),
    ("Nếu hồ sơ đã nộp trước ngày văn bản mới có hiệu lực, có được tiếp tục xử lý theo quy định cũ không?", "graph_traversal", "quy định chuyển tiếp"),
    ("Cơ sở nào xác định Cục trưởng có quyền bổ nhiệm cán bộ cấp phòng thuộc Cục Hóa chất?", "graph_traversal", "Cục Hóa chất"),
    ("Nếu cơ quan y tế cần vừa xử lý hồ sơ chuyển tiếp vừa sử dụng kinh phí ngân sách, có được đồng thời thực hiện không?", "hybrid_reasoning", "hồ sơ chuyển tiếp và ngân sách"),
    ("Nếu doanh nghiệp bảo hiểm muốn lưu trữ hồ sơ nghiệp vụ, có cần tuân thủ đồng thời Luật Lưu trữ không?", "hybrid_reasoning", "bảo hiểm và lưu trữ"),
    ("Nếu một địa phương triển khai hệ thống camera giao thông và cần huy động vốn ODA, cần tuân thủ những quy định nào?", "hybrid_reasoning", "camera giao thông và vốn ODA"),
    ("Nếu Sở Khoa học và Công nghệ vừa thực hiện thủ tục xây dựng vừa tuân thủ kế hoạch chuyển đổi số, cần căn cứ nhóm quy định nào?", "hybrid_reasoning", "xây dựng và chuyển đổi số"),
    ("Nếu chủ dự án nạo vét muốn đổ thải vào khu đất được UBND cấp xã chấp thuận, có được miễn trừ quy định môi trường không?", "hybrid_reasoning", "nạo vét, đất đai, môi trường"),
    ("Nếu tổ chức sử dụng lao động là người dân tộc thiểu số và bị thiệt hại do thiên tai, có được miễn giảm tiền thuê đất theo cả hai điều kiện không?", "hybrid_reasoning", "miễn giảm tiền thuê đất"),
    ("Nếu Bộ Công Thương dùng vốn vay ưu đãi cho nhiệm vụ thuộc lĩnh vực doanh nghiệp, có áp dụng đồng thời quy định ngân sách và chuyên ngành không?", "hybrid_reasoning", "vốn vay ưu đãi và doanh nghiệp"),
    ("Nếu Cục Hàng không Việt Nam triển khai dự án trong lĩnh vực đất đai và hồ sơ đã nộp trước văn bản mới, có xử lý theo quy định cũ không?", "hybrid_reasoning", "hàng không, đất đai, chuyển tiếp"),
    ("Nếu Bộ Y tế vừa làm thủ tục về hóa chất vừa tuân thủ kế hoạch chuyển đổi số, cần căn cứ nhóm quy định nào?", "hybrid_reasoning", "hóa chất và chuyển đổi số"),
    ("Nếu một cơ sở khám chữa bệnh ở Vĩnh Long chịu tác động của thủ tục mỹ phẩm bị bãi bỏ, có được thực hiện kỹ thuật mới không?", "hybrid_reasoning", "khám chữa bệnh và thủ tục mỹ phẩm"),
    ("Nếu người chủ trì hiệp đồng bay chọn tiêu chuẩn phân cách bay nhưng xung đột với thủ tục thừa phát lại, có áp dụng đồng thời không?", "hybrid_reasoning", "phân cách bay và thừa phát lại"),
    ("Nếu trường đại học tuyển sinh qua giao dịch điện tử theo thủ tục mới, có được pháp lý hỗ trợ không?", "hybrid_reasoning", "tuyển sinh và giao dịch điện tử"),
]


def _clarify_question(entity_hint: str = "văn bản hoặc điều luật") -> str:
    return f"Bạn muốn hỏi cụ thể về {entity_hint} nào? Vui lòng cung cấp tên, số hiệu hoặc điều khoản liên quan."


def _record(
    *,
    qid: int,
    query: str,
    history: str,
    expected_route: str,
    ambiguity_type: str,
    expected_behavior: str,
    gold_resolved_entity: str = "",
    gold_clarification_question: str = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "id": f"conv_{qid:04d}",
        "query": query,
        "history": history,
        "expected_route": expected_route,
        "ambiguity_type": ambiguity_type,
        "expected_behavior": expected_behavior,
        "gold_resolved_entity": gold_resolved_entity,
        "gold_clarification_question": gold_clarification_question,
        "notes": notes,
    }


def build_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    qid = 1

    for scenario in BASE_SCENARIOS:
        records.append(_record(
            qid=qid,
            query=scenario["query"],
            history=scenario["history"],
            expected_route=scenario["route"],
            ambiguity_type="answerable_with_history",
            expected_behavior="Resolve the demonstrative/pronoun from history and route to retrieval, not clarification.",
            gold_resolved_entity=scenario["entity"],
            notes="Paired resolving-history case.",
        ))
        qid += 1

        records.append(_record(
            qid=qid,
            query=scenario["query"],
            history="",
            expected_route="clarify",
            ambiguity_type="clarify_without_history",
            expected_behavior="Ask the user to specify the missing legal target before retrieval.",
            gold_clarification_question=_clarify_question(),
            notes="Paired empty-history case.",
        ))
        qid += 1

        records.append(_record(
            qid=qid,
            query=scenario["query"],
            history=scenario["irrelevant"],
            expected_route="clarify",
            ambiguity_type="irrelevant_history",
            expected_behavior="Detect that the existing history does not resolve the referent and ask a clarification question.",
            gold_clarification_question=_clarify_question(),
            notes="Paired irrelevant-history case.",
        ))
        qid += 1

        records.append(_record(
            qid=qid,
            query=scenario["query"],
            history=scenario["conflict"],
            expected_route="clarify",
            ambiguity_type="conflicting_history",
            expected_behavior="Ask which candidate document, article, agency, or issue the user means.",
            gold_clarification_question=_clarify_question("đối tượng pháp lý trong lịch sử hội thoại"),
            notes="Paired conflicting-history case.",
        ))
        qid += 1

    for query in MISSING_ENTITY_QUERIES:
        records.append(_record(
            qid=qid,
            query=query,
            history="",
            expected_route="clarify",
            ambiguity_type="missing_entity",
            expected_behavior="Ask for the missing legal document, article, entity, time period, or factual situation.",
            gold_clarification_question=_clarify_question(),
            notes="Semantic ambiguity; grammatically complete but missing retrieval target.",
        ))
        qid += 1

    for query in MULTI_INTERPRETATION_QUERIES:
        records.append(_record(
            qid=qid,
            query=query,
            history="",
            expected_route="clarify",
            ambiguity_type="multi_interpretation",
            expected_behavior="Ask which legal context or interpretation should be used.",
            gold_clarification_question=_clarify_question("bối cảnh pháp lý"),
            notes="Semantic ambiguity; multiple legal interpretations are plausible.",
        ))
        qid += 1

    for query, entity in DENSE_CONTROLS:
        records.append(_record(
            qid=qid,
            query=query,
            history="",
            expected_route="dense_retrieval",
            ambiguity_type="clear_dense_control",
            expected_behavior="Route to dense retrieval for direct legal lookup.",
            gold_resolved_entity=entity,
            notes="Clear non-clarify control.",
        ))
        qid += 1

    for query, route, entity in GRAPH_HYBRID_CONTROLS:
        records.append(_record(
            qid=qid,
            query=query,
            history="",
            expected_route=route,
            ambiguity_type="clear_graph_or_hybrid_control",
            expected_behavior="Route to graph or hybrid retrieval for relation-heavy legal reasoning.",
            gold_resolved_entity=entity,
            notes="Clear relation-heavy non-clarify control.",
        ))
        qid += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conversation-aware ambiguity benchmark")
    parser.add_argument("--output", default="evaluation/conversation_ambiguity_eval.json")
    args = parser.parse_args()

    records = build_records()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    counts = Counter(record["ambiguity_type"] for record in records)
    print(f"Wrote {len(records)} records to {output}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
