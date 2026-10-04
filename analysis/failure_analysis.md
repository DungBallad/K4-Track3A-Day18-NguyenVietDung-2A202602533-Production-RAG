# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Việt Dũng (2A202602533)  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

Test set: 20 câu hỏi (`test_set.json`), LLM sinh câu trả lời `gpt-4o-mini`. Kết quả lấy từ `reports/naive_baseline_report.json` và `reports/ragas_report.json`.

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7617 | **0.8304** | +0.0688 |
| Answer Relevancy | 0.7582 | **0.7679** | +0.0097 |
| Context Precision | 0.9500 | **0.9042** | -0.0458 |
| Context Recall | 0.9250 | **0.9083** | -0.0167 |

- Naive baseline = paragraph chunking (57 chunks) + dense-only (bge-m3), top-3, không rerank, không enrichment.
- Production = hierarchical chunking (104 child chunks) + M5 enrichment combined (1 call/chunk) + Hybrid BM25/Dense + RRF + cross-encoder `bge-reranker-v2-m3` (top-3).
- Cả 4 metrics của Production đều ≥ 0.75.

**Nhận xét:** Faithfulness tăng rõ nhất (+0.07) vì context sau rerank sạch hơn và enrichment thêm ngữ cảnh tài liệu vào mỗi chunk, LLM ít phải "đoán". Context precision/recall giảm nhẹ vì child chunk 256 ký tự nhỏ hơn paragraph chunk: với các câu hỏi multi-hop, top-3 child chunk đôi khi không phủ đủ thông tin (xem #5). Pipeline hiện index child chunk nhưng **chưa trả về parent** khi generate — đây là cải tiến rõ ràng nhất cần làm tiếp.

## Latency breakdown (từ `reports/run_log.txt`)

| Bước | Thời gian |
|------|-----------|
| M1 Chunking (26 docs → 104 chunks) | 0.1s |
| M5 Enrichment (104 API calls) | 290.0s (~2.8s/chunk) |
| M2 Indexing BM25 + Dense (bge-m3, CPU) | 59.0s |
| M3 Load reranker | ~0s (lazy load, ~5s ở query đầu) |
| RAGAS eval Production (80 metric calls) | 106.9s |
| Tổng `main.py` (baseline + production + so sánh) | 861.6s |

Enrichment là bottleneck offline lớn nhất → nên chạy song song (async/batch) và cache kết quả theo hash của chunk.

---

## Bottom-5 Failures

### #1
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), đổi mỗi 120 ngày; chính sách cũ 90 ngày đã bị thay thế.
- **Got:** "Có hai chu kỳ… 90 ngày và 120 ngày. Không tìm thấy thông tin cụ thể để xác định…"
- **Worst metric:** answer_relevancy = 0.00 (avg 0.50)
- **Error Tree:** Output sai → Context đúng? **Có, nhưng lẫn cả v1.0 lẫn v2.0** → Query OK? Có → lỗi ở bước **retrieval filtering + prompt** (không xử lý xung đột phiên bản).
- **Root cause:** `mat_khau_v1.md` (trạng thái "ĐÃ THAY THẾ bởi v2.0") và `mat_khau_v2.md` đều được retrieve. Prompt không có quy tắc ưu tiên phiên bản mới, nên LLM trả lời lấp lửng.
- **Suggested fix:** Trích `version`, `effective_date`, `status` vào metadata (M5 `extract_metadata`) → filter bỏ tài liệu có `status=superseded` hoặc khi có nhiều phiên bản thì chỉ giữ bản mới nhất; thêm vào system prompt: "Nếu có nhiều phiên bản, dùng phiên bản hiện hành mới nhất".

### #2
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Chỉ được tham gia BHXH bắt buộc.
- **Got:** "Không tìm thấy."
- **Worst metric:** answer_relevancy = 0.00 (avg 0.75 → 3 metric còn lại = 1.0)
- **Error Tree:** Output sai → Context đúng? **Có** (context precision/recall = 1.0) → Query OK? Có → lỗi ở bước **generation**.
- **Root cause:** Câu trả lời nằm trong `thu_viec.md` ("…chưa được hưởng gói bảo hiểm sức khỏe PVI"), context đã có, nhưng prompt quá cứng ("Nếu không có → nói 'Không tìm thấy'") khiến LLM chọn từ chối khi thông tin được diễn đạt dạng phủ định/gián tiếp, cộng thêm chunk `bao_hiem_suc_khoe.md` ("cho tất cả nhân viên chính thức") gây nhiễu.
- **Suggested fix:** Sửa prompt: yêu cầu đọc kỹ toàn bộ context, trả lời Có/Không kèm trích dẫn câu nguồn trước khi kết luận "Không tìm thấy"; dùng `temperature=0`.

### #3
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt (5–50 triệu), phòng CNTT xác nhận cấu hình, cần ≥ 3 báo giá (> 10 triệu).
- **Got:** "…cần xác nhận của phòng CNTT… phê duyệt từ **trưởng phòng**…"
- **Worst metric:** faithfulness = 0.50 (avg 0.74)
- **Error Tree:** Output sai → Context đúng? **Thiếu một phần** (bảng thẩm quyền phê duyệt) → Query OK? Câu hỏi multi-part → lỗi ở bước **chunking**.
- **Root cause:** Hierarchical chunking cắt child theo câu/dòng 256 ký tự nên **bảng markdown thẩm quyền** trong `mua_sam.md` bị tách khỏi phần "Lưu ý đặc biệt" về CNTT. Top-3 chỉ có phần CNTT → LLM bịa "trưởng phòng" (hallucination) và bỏ sót yêu cầu 3 báo giá.
- **Suggested fix:** Dùng `chunk_structure_aware()` cho tài liệu có bảng (giữ nguyên section) hoặc trả về **parent chunk** sau khi retrieve child; tăng `RERANK_TOP_K` lên 5 cho câu hỏi nhiều ý.

### #4
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày, phí 2%/tháng × 15 triệu = 300.000đ/tháng, pro-rata ≈ 50.000đ cho 5 ngày.
- **Got:** "…phạt 300.000 VNĐ" (tính tròn 1 tháng).
- **Worst metric:** faithfulness = 0.375 (avg 0.75)
- **Error Tree:** Output sai → Context đúng? **Có** (`tam_ung.md`: hạn 15 ngày, phí 2%/tháng) → Query OK? Có → lỗi ở bước **generation (suy luận số học)**.
- **Root cause:** LLM tự suy diễn "tính theo tháng nên làm tròn 1 tháng" — một khẳng định không có trong context → RAGAS đánh faithfulness thấp. Câu hỏi yêu cầu tính toán mà prompt không hướng dẫn cách xử lý.
- **Suggested fix:** Prompt chain-of-thought có ràng buộc: "Chỉ dùng số liệu trong context, nêu rõ giả định khi phải tính toán"; hoặc route câu hỏi tính toán sang tool/calculator; `temperature=0`.

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép; lương Senior (P3–P4) 20–35 triệu/tháng.
- **Got:** "…18 ngày phép năm (15 + 3)…" — **thiếu phần lương**.
- **Worst metric:** context_recall = 0.50 (avg 0.75)
- **Error Tree:** Output thiếu → Context đúng? **Thiếu** (chỉ có `nghi_phep_nam_v2024.md`, không có `bang_luong_2024.md`) → Query OK? **Không** — câu hỏi ghép 2 ý, 1 query vector bị kéo về chủ đề nghỉ phép → lỗi ở bước **query / retrieval**.
- **Root cause:** Câu hỏi multi-hop cần 2 tài liệu. Hybrid search + rerank top-3 đều ưu tiên các chunk nghỉ phép (khớp cả từ khóa "thâm niên", "nghỉ phép"), chunk bảng lương bị đẩy ra khỏi top-3.
- **Suggested fix:** Query decomposition (tách thành "số ngày phép thâm niên 9 năm" và "khung lương Senior"), retrieve riêng rồi merge; hoặc tăng `RERANK_TOP_K` lên 5 kèm diversity theo `source`.

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Bao lâu phải đổi mật khẩu một lần?" (#1 — câu tệ nhất, avg 0.50)

**Error Tree walkthrough:**
1. Output đúng? → **Không.** Trả lời lấp lửng giữa 90 và 120 ngày, answer_relevancy = 0.
2. Context đúng? → **Có nhưng nhiễu.** Retrieval lấy cả `mat_khau_v1.md` (đã bị thay thế) và `mat_khau_v2.md` (hiện hành). Recall cao nhưng chứa thông tin mâu thuẫn.
3. Query rewrite OK? → **Có.** Query ngắn, rõ, không cần rewrite.
4. Fix ở bước: **Retrieval filtering (metadata) + Prompt.** Đây là lỗi "version conflict" điển hình của RAG doanh nghiệp: hybrid search và reranker chỉ đo độ liên quan ngữ nghĩa, không biết tài liệu nào còn hiệu lực. Cần metadata `status/effective_date` (M5) để lọc trước khi đưa vào LLM, và quy tắc "ưu tiên bản mới nhất" trong prompt.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Trả về parent chunk (2048) thay vì child (256) khi generate → sửa #3, #5 và kéo context recall lên lại.
- Metadata filter theo version/status + ưu tiên bản mới nhất → sửa #1 (và câu MFA cũng bị nhiễu bởi v1.0).
- Viết lại system prompt: `temperature=0`, trích dẫn câu nguồn, nêu giả định khi tính toán → sửa #2, #4.
- Query decomposition cho câu hỏi nhiều ý, `RERANK_TOP_K=5`.
- Chạy enrichment song song + cache để giảm 290s xuống < 60s.
