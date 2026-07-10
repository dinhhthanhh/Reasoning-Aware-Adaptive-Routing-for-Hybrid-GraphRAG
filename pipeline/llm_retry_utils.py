"""
llm_retry_utils.py
===================
Retry + rate-limit helper cho lời gọi LLM API. Mục tiêu: loại bỏ tình trạng
~80/600 câu (test_0520 - test_0599, ~13%) bị trả về CHUỖI LỖI

    "Lỗi khi tạo câu trả lời: OpenAI generation failed after 3 attempts:
     HTTP 429 Too Many Requests: ..."

làm GIÁ TRỊ PREDICTION trong file kết quả -> kéo méo toàn bộ Token F1,
Routing Acc, BERTScore, v.v. của lần chạy gần nhất.

NGUYÊN NHÂN GỐC:
  - Prompt v1 (verbatim toàn bộ Điều) làm output mỗi câu dài hơn nhiều
    lần -> tốn nhiều token/giây hơn -> chạm rate limit (429) thường
    xuyên hơn, đặc biệt về cuối run khi quota gần hết.
  - eval runner hiện tại bắt exception sau 3 lần retry rồi GHI CHUỖI
    THÔNG BÁO LỖI vào trường "prediction" -> bị tính như một câu trả lời
    hợp lệ (sai!) trong mọi script tổng hợp.

CÁCH DÙNG (tích hợp vào pipeline/hybrid_pipeline.py):

    from pipeline.llm_retry_utils import RateLimiter, call_llm_with_backoff

    # Tạo 1 limiter DÙNG CHUNG cho cả pipeline (1 instance, không tạo mới
    # mỗi lần gọi), vd trong __init__ của HybridPipeline:
    self._llm_limiter = RateLimiter(min_interval_sec=1.5)

    # ... trong generate_answer():
    try:
        response = call_llm_with_backoff(
            fn=lambda: self.llm_client.chat(messages=messages, max_tokens=1024),
            limiter=self._llm_limiter,
            max_retries=8,
            base_delay=2.0,
        )
    except Exception as exc:
        # KHÔNG trả về str(exc) như một câu trả lời!
        # Ghi log + để None/sentinel cho eval runner biết cần chạy lại.
        logger.error("LLM call failed after retries for query=%r: %s", query, exc)
        return None   # eval runner: nếu prediction is None -> SKIP, không
                       # ghi vào predictions.jsonl, để lần chạy sau resume.

CÁCH DÙNG (resume eval runner sau khi bị gián đoạn — xem load_done_ids /
append_result ở cuối file).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

# -----------------------------------------------------------------------
# Nhận diện lỗi "tạm thời, nên retry" (rate limit, quota, overload, timeout)
# -----------------------------------------------------------------------
_RETRYABLE_MARKERS = (
    "429",
    "too many requests",
    "rate limit",
    "rate_limit",
    "quota",
    "overloaded",
    "503",
    "502",
    "timeout",
    "connection",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


# -----------------------------------------------------------------------
# RateLimiter — đảm bảo khoảng cách tối thiểu giữa 2 lần gọi API liên tiếp
# để KHÔNG TỰ ĐÂM vào rate limit ngay từ đầu (phòng bệnh hơn chữa bệnh).
#
# Giá trị khởi điểm gợi ý: 1.0 - 2.0s (≈ 30-60 request/phút). Nếu vẫn còn
# 429 sau khi áp dụng, tăng dần (vd 3.0s). Nếu provider cho throughput cao
# hơn và muốn chạy nhanh, có thể giảm xuống (vd 0.5s) rồi quan sát log
# [RETRY] có xuất hiện không.
# -----------------------------------------------------------------------
class RateLimiter:
    """Giới hạn tối thiểu `min_interval_sec` giây giữa 2 lần gọi liên tiếp."""

    def __init__(self, min_interval_sec: float = 1.5):
        self.min_interval_sec = min_interval_sec
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_interval_sec - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


# -----------------------------------------------------------------------
# call_llm_with_backoff — exponential backoff + jitter, retry nhiều hơn 3
# lần. QUAN TRỌNG: nếu hết retry, RAISE exception gốc — KHÔNG trả về một
# chuỗi mô tả lỗi để dùng làm prediction.
# -----------------------------------------------------------------------
def call_llm_with_backoff(
    fn: Callable[[], T],
    limiter: RateLimiter | None = None,
    max_retries: int = 8,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> T:
    """
    Gọi fn() với rate limiting + exponential backoff + jitter.

    Args:
        fn          : callable không nhận argument, vd
                       `lambda: self.llm_client.chat(messages=messages)`
        limiter     : RateLimiter dùng chung cho pipeline (có thể None)
        max_retries : số lần thử tối đa cho lỗi "retryable" (mặc định 8,
                       thay vì 3 như cũ — mỗi lần backoff tăng dần nên
                       8 lần vẫn kết thúc trong vài phút, không treo vô hạn)
        base_delay  : delay (giây) cho lần retry đầu tiên
        max_delay   : delay tối đa (giây) cho mỗi lần retry

    Raises:
        Exception cuối cùng nếu hết max_retries hoặc lỗi không "retryable"
        (vd lỗi cú pháp prompt, auth lỗi...). Caller PHẢI bắt exception
        này riêng và xử lý như "câu hỏi cần chạy lại sau", KHÔNG ghi vào
        predictions.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        if limiter is not None:
            limiter.wait()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - cần bắt mọi lỗi provider
            last_exc = exc
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)  # jitter tránh "thundering herd"
            print(
                f"[RETRY {attempt}/{max_retries}] {type(exc).__name__}: "
                f"{exc}. Chờ {delay:.1f}s rồi thử lại..."
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# -----------------------------------------------------------------------
# Checkpoint helpers — cho phép RESUME eval runner sau khi bị gián đoạn,
# KHÔNG cần chạy lại toàn bộ 600 câu mỗi lần.
# -----------------------------------------------------------------------
ERROR_PREDICTION_MARKERS = (
    "Lỗi khi tạo câu trả lời:",
    "Lỗi khi tổng hợp câu trả lời",
    "OpenAI generation failed",
    "HTTP 429",
    "Too Many Requests",
)


def is_error_prediction(text: str) -> bool:
    """Nhận diện các prediction thực chất là THÔNG BÁO LỖI bị lưu nhầm."""
    if not text:
        return False
    return any(marker in text for marker in ERROR_PREDICTION_MARKERS)


def load_done_ids(jsonl_path: str | Path) -> set[str]:
    """
    Đọc file predictions JSONL hiện có, trả về set các id ĐÃ XONG HỢP LỆ.

    Các id có prediction là "thông báo lỗi" (is_error_prediction == True)
    KHÔNG được tính vào set này -> coi như CHƯA XONG, sẽ được chạy lại
    nếu eval runner dùng load_done_ids() để quyết định skip/không-skip.
    """
    path = Path(jsonl_path)
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(obj.get("id", obj.get("question_id", "")))
            pred = str(obj.get("prediction", obj.get("answer", "")))
            if qid and not is_error_prediction(pred):
                done.add(qid)
    return done


def append_result(jsonl_path: str | Path, result: dict) -> None:
    """Append 1 dòng JSON vào file predictions (tạo file/folder nếu chưa có)."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
