# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Việt Dũng (2A202602533)  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Tách câu bằng regex, embed bằng `all-MiniLM-L6-v2`, gom các câu liền kề có cosine ≥ 0.85 (`SEMANTIC_THRESHOLD`). Threshold cao nên tiếng Việt bị cắt khá vụn (model MiniLM là model tiếng Anh), vì vậy pipeline production chọn hierarchical thay vì semantic. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent 2048 / child 256 ký tự, child giữ `parent_id`. 26 tài liệu → **104 child chunks** (so với 57 paragraph chunks của baseline). Child nhỏ giúp match chính xác nhưng làm context precision/recall giảm nhẹ (0.95→0.90, 0.925→0.908) vì chưa trả parent khi generate. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Split theo header `#`–`###`, lưu `section` vào metadata. Phù hợp cho `mua_sam.md` có bảng thẩm quyền — failure #3 cho thấy nên dùng chiến lược này cho tài liệu có bảng. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép") nên phải `replace("_", " ")` để query và corpus cùng token. RRF `1/(k+rank+1)`, k=60, chỉ dùng thứ hạng nên không cần chuẩn hoá điểm BM25 (không giới hạn) với cosine (0–1). BM25 bắt tốt số liệu/từ khoá chính xác ("MFA", "PVI"), dense bắt được câu hỏi diễn đạt khác. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` chấm từng cặp (query, doc) cho 20 candidate sau RRF → giữ top-3. Model được cache trong `_MODEL_CACHE` nên chỉ load 1 lần (~5s ở query đầu). Rerank giúp faithfulness tăng +0.07, nhưng top-3 quá chặt với câu hỏi multi-hop (failure #5). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: Faithfulness 0.830, Answer Relevancy **0.768 (thấp nhất)**, Context Precision 0.904, Context Recall 0.908. Answer relevancy thấp vì LLM trả "Không tìm thấy" hoặc trả lời lấp lửng khi context có xung đột phiên bản (mật khẩu v1/v2). `failure_analysis()` map metric tệ nhất → diagnosis/fix theo Diagnostic Tree. |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()` (combined mode) | 1 call `gpt-4o-mini`/chunk sinh summary + hypothesis questions + context prepend + metadata. 104 chunks mất 290s — bước chậm nhất pipeline. Context prepend giúp child chunk ngắn vẫn mang tên tài liệu/chủ đề, giảm retrieval failure khi chunk bị cắt khỏi tiêu đề. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi 1 — Tests M2 fail khi chạy `check_lab.py`:**
  - Exact error: `ModuleNotFoundError: No module named 'rank_bm25'` (2 test `test_bm25_search`, `test_bm25_relevant_first`).
  - Debug: `requirements.txt` đã có `rank-bm25>=0.2.2`, nên kiểm tra `sys.executable` → script đang chạy bằng Python hệ thống (`Python311\python.exe`) chứ không phải `.venv`. Chạy lại bằng `.venv\Scripts\python.exe -m pytest tests/` → **37/37 passed**.
  - Bài học: luôn activate virtualenv trước khi chạy test/grader; `check_lab.py` gọi `sys.executable` nên dùng đúng interpreter đang chạy nó.

- **Lỗi 2 — PDF trong `data/` không đọc được:**
  - Exact message: `Ignoring wrong pointing object 11 0 (offset 0)` (cảnh báo pypdf), và `BCTC.pdf`, `Nghi_dinh_so_13-2023...pdf` trả về text rỗng.
  - Debug: `pypdf.extract_text()` trả chuỗi rỗng → đây là PDF scan ảnh, không có text layer. Thêm kiểm tra trong `load_documents()` để bỏ qua và log `⚠️ Bỏ qua ...: PDF scan ảnh, không có text layer (cần OCR)` thay vì index chunk rỗng làm nhiễu BM25.
  - Kiến thức thiếu: phân biệt PDF text vs PDF scan; hướng bổ sung là OCR (Tesseract/`pytesseract`, hoặc docling/unstructured) trước khi chunk.

- **Lỗi 3 — Tải model HuggingFace (bge-m3, bge-reranker-v2-m3) chậm/treo:**
  - Hiện tượng: lần đầu chạy pipeline đứng lâu ở bước tải weights, kèm cảnh báo `Warning: You are sending unauthenticated requests to the HF Hub`.
  - Cách giải quyết: trong `config.py` đặt `HF_HUB_DISABLE_XET=1` và `HF_ENDPOINT=https://hf-mirror.com` để tải qua mirror; thêm `_MODEL_CACHE` ở M3 để không load lại reranker mỗi lần khởi tạo.

- **Lỗi 4 — RAGAS trả NaN cho một số câu:**
  - Hiện tượng: `result.to_pandas()` có giá trị `NaN` khi LLM judge lỗi/timeout, làm `float(...)` và trung bình bị sai.
  - Cách giải quyết: trong `evaluate_ragas()` kiểm tra `x != x` (NaN) → thay bằng 0.0, bọc toàn bộ trong `try/except` để pipeline vẫn ghi report thay vì crash.

- **Kiến thức còn thiếu & cách bổ sung:** cơ chế chấm của từng metric RAGAS (vì sao "Không tìm thấy." có faithfulness 1.0 nhưng answer relevancy 0) → đọc lại docs RAGAS v0.1 và đối chiếu từng câu trong `reports/ragas_report.json`.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý hỏi đáp chính sách & quy trình nội bộ (HR / IT / Tài chính) cho doanh nghiệp

#### 1. Hiện trạng
- **Pipeline hiện tại:** Naive RAG — chia đoạn theo paragraph, dense search (1 embedding model), top-3 đưa thẳng vào LLM.
- **Vấn đề / Bottlenecks đang gặp:** tài liệu có nhiều phiên bản (cũ/mới) gây trả lời mâu thuẫn; câu hỏi nhiều ý chỉ được trả lời một nửa; tài liệu PDF scan không đọc được; chưa có bộ đo chất lượng nên không biết sửa có cải thiện hay không.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Hierarchical (child 256 để retrieve, trả parent 2048 khi generate) cho văn bản thường; structure-aware cho tài liệu có header/bảng để không cắt rời bảng thẩm quyền.
2. **Search retrieval:** Hybrid BM25 (underthesea segmentation) + Dense bge-m3 + RRF — BM25 cần cho mã số, số tiền, tên viết tắt (MFA, PVI); dense cho câu hỏi diễn đạt tự nhiên. Thêm metadata filter `status/effective_date` để loại tài liệu đã bị thay thế.
3. **Reranking:** Có — `bge-reranker-v2-m3` (đa ngôn ngữ, hỗ trợ tiếng Việt), top-20 → top-5, ưu tiên đa dạng nguồn cho câu hỏi multi-hop.
4. **Evaluation:** RAGAS 4 metrics trên golden set ~50 câu (mở rộng từ 20 câu của lab, thêm nhóm câu version-conflict và multi-hop); chạy lại mỗi lần đổi pipeline, ngưỡng chấp nhận mọi metric ≥ 0.80.
5. **Enrichment:** Combined single-call (summary + hypothesis questions + contextual prepend + metadata) — metadata version/ngày hiệu lực là phần quan trọng nhất để xử lý xung đột phiên bản; cache kết quả theo hash chunk và chạy async.

#### 3. Timeline triển khai
- **Tuần 1:** Xây golden test set 50 câu + chạy RAGAS cho pipeline hiện tại làm baseline; thêm OCR cho PDF scan.
- **Tuần 2:** Hierarchical + structure-aware chunking, trả parent khi generate; hybrid search BM25 + dense + RRF. Đo lại RAGAS.
- **Tuần 3:** Reranker + enrichment combined (async, có cache) + metadata filter theo version. Viết lại prompt (temperature 0, trích dẫn nguồn).
- **Tuần 4:** Query decomposition cho câu hỏi nhiều ý, latency breakdown từng bước, tối ưu và báo cáo so sánh trước/sau.
