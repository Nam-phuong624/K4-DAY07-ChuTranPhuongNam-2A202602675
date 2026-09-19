# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Chử Trần Phương Nam
**MSSV:** 2A202602675
**Nhóm:** GO HOME
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là góc giữa hai vector embedding trong không gian đa chiều rất nhỏ, phản ánh hai đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa và chủ đề nội dung, bất kể sự khác biệt về độ dài câu hay từ ngữ cụ thể.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên có thể mượn tối đa 5 cuốn sách tại thư viện trường."
- Câu B: "Thư viện đại học cho phép mỗi người học mượn không quá 5 tài liệu."
- Tại sao tương đồng: Cả hai câu cùng mô tả chính xác một quy định về số lượng tài liệu tối đa được mượn ở thư viện trường với các từ ngữ đồng nghĩa thay thế (sinh viên/người học, cuốn sách/tài liệu).

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Thủ tục xin cấp giấy chứng nhận sinh viên tạm hoãn nghĩa vụ quân sự."
- Câu B: "Cơ chế quang hợp và trao đổi chất ở lá cây nhiệt đới."
- Tại sao khác: Hai câu thuộc hai lĩnh vực tri thức hoàn toàn tách biệt (quy chế hành chính học vụ đối lập với sinh học thực vật), không có điểm chung về ngữ cảnh ngữ nghĩa.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị chi phối bởi độ dài vector (thường tương quan với độ dài văn bản hoặc tần suất từ lặp lại), khiến hai câu cùng ý nghĩa nhưng khác độ dài bị xem là xa nhau; trong khi đó, độ tương tự cosine chuẩn hóa độ dài và chỉ đo góc định hướng, giúp tập trung thuần túy vào mối tương quan ngữ nghĩa trừu tượng.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Áp dụng công thức: `số lượng chunk = làm_tròn_lên((độ_dài_tài_liệu - độ_chồng_chéo) / (kích_thước_chunk - độ_chồng_chéo))`
> `số lượng chunk = ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.111...) = 23`
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi `overlap=100`, phép tính là `ceil((10,000 - 100) / (500 - 100)) = ceil(9,900 / 400) = ceil(24.75) = 25 chunks` (tăng thêm 2 chunks). Chúng ta muốn độ chồng chéo nhiều hơn để giảm thiểu nguy cơ mất ngữ cảnh tại các điểm cắt (boundary cut), đảm bảo các mệnh đề, liên từ hoặc thực thể ngữ nghĩa quan trọng không bị chia cắt vụn vỡ giữa hai chunk liền kề.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy `re.split(r'(?<=[.!?])\s+|\.\n+', text.strip())` để nhận diện các ranh giới kết thúc câu mà vẫn giữ nguyên được nội dung câu. Xử lý các ngoại lệ (edge case) như văn bản rỗng, chuỗi chỉ toàn khoảng trắng hoặc văn bản ngắn chưa đủ dấu câu, sau đó gom các câu hợp lệ thành từng nhóm tối đa `max_sentences_per_chunk` và loại bỏ khoảng trắng dư thừa.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán duyệt qua danh sách dấu phân cách theo thứ tự ưu tiên giảm dần `["\n\n", "\n", ". ", " ", ""]`. Trường hợp cơ sở (base case) là khi độ dài văn bản hiện tại nhỏ hơn hoặc bằng `chunk_size` (trả về chính nó) hoặc khi danh sách dấu phân cách đã cạn kiệt (cắt đều theo ký tự). Với từng dấu phân cách, chia nhỏ và gom các mẩu nhỏ lại nếu chưa vượt ngưỡng `chunk_size`, nếu gặp một mẩu con quá dài thì lập tức đệ quy với tập phân cách kế tiếp.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mỗi tài liệu được chuẩn hóa thành một record trong `self._store` gồm `id`, `content`, `metadata` (bổ sung `doc_id`), và vector `embedding` được tạo từ `self._embedding_fn`. Khi tìm kiếm (`search`), nhúng chuỗi query thành vector rồi tính tích vô hướng (dot product) với từng embedding đã lưu (vì embedding đã chuẩn hóa nên dot product tương đương cosine similarity), sau đó sắp xếp giảm dần theo điểm số để lấy `top_k` kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Áp dụng cơ chế tiền lọc (pre-filtering): quét qua `self._store` và chỉ giữ lại các record thỏa mãn toàn bộ các cặp key-value trong `metadata_filter` trước khi tính điểm tương tự để tối ưu tài nguyên tính toán. Đối với hàm `delete_document`, tạo lại danh sách bản ghi loại trừ các record có `id` hoặc `metadata['doc_id']` trùng với `doc_id` được yêu cầu xóa, trả về `True` nếu kích thước store giảm đi.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Thực hiện bước truy xuất (retrieval) bằng cách gọi `self.store.search(question, top_k=top_k)`. Nối ghép các đoạn nội dung `content` thu được thành chuỗi ngữ cảnh context phân tách bằng ký tự xuống dòng kép, sau đó đưa vào cấu trúc prompt RAG rõ ràng: `Context:\n{context}\n\nQuestion: {question}\nAnswer:` và gọi `self.llm_fn` để sinh câu trả lời dựa trên ngữ cảnh đã tìm được.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/namphuong/Desktop/vin_lab/K4-Day07-ChuTranPhuongNam-2A202602675
plugins: requests-mock-1.12.1, anyio-4.13.0
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.05s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên có thể mượn tối đa 5 cuốn sách tại thư viện trường. | Người học được phép mượn không quá 5 đầu sách ở thư viện đại học. | cao | -0.0378 | Không (với Mock) |
| 2 | Hạn chót đóng học phí học kỳ 1 là ngày 30 tháng 10. | Thời hạn hoàn thành nghĩa vụ học phí kỳ 1 kết thúc vào ngày 30/10. | cao | 0.2905 | Một phần |
| 3 | Quy định đăng ký môn học và xét tốt nghiệp ra trường. | Thực đơn trưa hôm nay tại căng tin gồm cơm gà và canh chua. | thấp | 0.0389 | Đúng |
| 4 | Trí tuệ nhân tạo và học máy đang thay đổi thế giới công nghệ. | Các thuật toán machine learning và AI định hình lại nền công nghiệp hiện đại. | cao | -0.0099 | Không (với Mock) |
| 5 | Hướng dẫn thủ tục xin cấp giấy xác nhận sinh viên tạm hoãn nghĩa vụ quân sự. | Hệ thống năng lượng mặt trời giúp tiết kiệm điện năng cho hộ gia đình. | thấp | -0.0218 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là ở các cặp câu đồng nghĩa (Cặp 1 và Cặp 4), trực giác con người và mô hình embedding ngữ nghĩa thực thụ (như MiniLM hay OpenAI) sẽ cho điểm tương đồng rất cao (> 0.85), nhưng trên `MockEmbedder`, điểm số lại rơi vào khoảng xấp xỉ 0 (-0.0378 và -0.0099). Điều này cho thấy sự khác biệt cốt lõi: mô hình embedding giả lập (dựa trên thuật toán băm MD5 deterministic) xem các từ ngữ khác nhau là các chuỗi ký tự ngẫu nhiên trực giao với nhau trong không gian; trong khi đó, embedding học sâu (deep learning embeddings) mới thực sự ánh xạ được ngữ nghĩa tiềm ẩn (semantic latent space) để nhận ra tính tương đồng của các từ đồng nghĩa và ngữ cảnh tương đương.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chiến lược cá nhân của tôi là: **`RecursiveChunker(chunk_size=400)`** chạy với backend **OpenAI Semantic Embeddings (`text-embedding-3-small`)** và Agent LLM **`gpt-4o-mini`**.

Kết quả đo đạc thực nghiệm trên bộ 5 câu hỏi benchmark chuẩn của nhóm bằng file `bench.py` (lưu tại `ket_qua_benchmark.txt`):

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên năm thứ mấy và cần đạt kết quả học tập thế nào để đủ điều kiện xét học bổng EVN? | `hoc-bong-evn#0`: Thông báo học bổng EVN 2025-2026, đối tượng sinh viên năm 3, GPA giỏi... | 0.6312 | **CÓ (Top-1)** | Sinh viên năm thứ 3 và cần đạt kết quả học tập loại giỏi trở lên để đủ điều kiện xét học bổng EVN. |
| 2 | Thời gian nghỉ Tết Nguyên đán năm học 2025-2026 của sinh viên chính quy kéo dài từ ngày nào đến ngày nào? | `ke-hoach-dao-tao-nam-hoc#4`: Bảng lịch trình công tác học kỳ I và mốc nghỉ Tết... | 0.5844 | **CÓ (Top-1)** | Thời gian nghỉ Tết Nguyên đán năm học 2025-2026 của sinh viên chính quy kéo dài từ ngày 09/02 đến 22/02/2026. |
| 3 | Phương thức 2 của Trường ĐH Khoa học Tự nhiên năm 2026 áp dụng nhân hệ số 2 môn Toán cho những ngành nào? | `phuong-thuc-xet-tuyen-hus#0`: 6 phương thức tuyển sinh HUS, lưu ý nhân đôi Toán... | 0.6919 | **CÓ (Top-1)** | Phương thức 2 áp dụng nhân hệ số 2 môn Toán cho các ngành: Toán học, Toán tin, Khoa học máy tính và thông tin, Khoa học dữ liệu. |
| 4 | Chương trình trao đổi sinh viên tại Đại học Osaka kỳ Xuân 2027 có bao nhiêu chỉ tiêu và yêu cầu điểm GPA tối thiểu là bao nhiêu? | `trao-doi-sinh-vien-osaka#0`: Thông báo chỉ tiêu và điều kiện GPA trao đổi ĐH Osaka... | 0.6842 | **CÓ (Top-1)** | Chương trình trao đổi sinh viên tại Đại học Osaka kỳ Xuân 2027 có 03 chỉ tiêu, yêu cầu điểm GPA tối thiểu là 3,2/4,0. |
| 5 | [Filter: student] Số lượng và danh mục các chương trình đào tạo chuẩn và đặc thù của Trường Đại học Công nghệ là gì? | `chuong-trinh-dao-tao-dai-hoc#0`: Bảng danh mục chương trình đào tạo ĐH Công nghệ... | 0.6959 | **CÓ (Top-1)** | Gồm 5 chương trình chuẩn (CNTT, Kỹ thuật máy tính, Robot, AI...) và 5 chương trình đặc thù CLC. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5** (Độ chính xác tuyệt đối: **Top-1 Accuracy: 100%**, **Top-3 Accuracy: 100%**).

**Phân tích nguyên nhân kết quả vượt trội khi dùng OpenAI Embeddings:**
> Khác biệt hoàn toàn so với mô hình băm giả lập `MockEmbedder`, mô hình học sâu `text-embedding-3-small` thực sự ánh xạ được ngữ nghĩa tiếng Việt vào không gian đa chiều (1536 chiều). Các từ khóa học vụ như *"nghỉ Tết"*, *"xét học bổng"*, *"nhân hệ số 2"* hay *"chương trình đào tạo"* được khớp trực tiếp với đúng phân đoạn tài liệu quy chế liên quan với điểm cosine similarity rất cao (từ `0.5844` đến `0.6959`). Đồng thời, Agent `gpt-4o-mini` đọc ngữ cảnh chính xác và trích xuất câu trả lời chuẩn xác 100% đúng với gold answer.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Việc kết hợp giữa chiến lược phân đoạn văn bản thông minh (`RecursiveChunker` tôn trọng ranh giới dòng của bảng markdown) với một embedding model ngữ nghĩa thực thụ là yếu tố quyết định thành công của một hệ thống RAG thực tế. Nếu chunking sai (làm rách bảng), dù embedding model có mạnh thì ngữ cảnh đưa vào LLM vẫn bị đứt gãy.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |

