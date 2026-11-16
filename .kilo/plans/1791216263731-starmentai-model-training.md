# Kế hoạch nâng chất lượng & tối ưu StarmentAI

## 1. Mục tiêu

Cải thiện Qwen3 0.6B trong StarmentAI theo 4 hướng:

1. **Đúng ngữ cảnh**: không mất/lố token, bám hội thoại, nhớ thông tin trong `num_ctx`.
2. **Đúng thông tin**: giảm bịa, biết nói không biết, tiếng Việt tự nhiên, giải thích kỹ thuật chính xác.
3. **Đúng phong cách StarmentAI**: thẳng, sắc, châm biếm thông minh; nghiêm túc với chủ đề nhạy cảm; không công kích cá nhân.
4. **Nhẹ hơn**: giảm CPU/RAM/latency trên máy hiện tại, không hy sinh chất lượng.

Nguyên tắc bắt buộc:

- Không kết luận nguyên nhân khi chưa có baseline/đo đạc.
- Không tạo dữ liệu giả để tăng số lượng.
- Không huấn luyện trên 3 mẫu hiện có.
- Mọi thay đổi model phải có bản baseline, eval trước/sau và đường rollback.

## 2. Hiện trạng đã xác minh

### 2.1 Code/test

- `pytest -q`: **98 passed** (đã chạy với cache tắt).
- Các thay đổi Phase 1–2 đang nằm trong working tree, chưa commit.
- `core/__pycache__/*.pyc` bị xóa trong diff; không ảnh hưởng runtime.
- `data/starment_dataset.jsonl` đang untracked, chỉ có **3 mẫu**.
- `training/` tồn tại nhưng rỗng.

### 2.2 Cấu hình hiện tại

- Model: `qwen3:0.6b`, custom `starmentai:latest` (Ollama 0.40.16).
- `Modelfile`: `temperature 0.6`, `top_p 0.9`, `top_k 40`, `repeat_penalty 1.1`, `num_ctx 2048`, `num_thread 2`, `num_predict 512`.
- `config.py`: `system_prompt_tokens=256`, `style_reminder` dài 431 ký tự, `max_history_messages=20`, `keep_alive=30s`.
- `OllamaClient` đã bỏ double system prompt; chỉ còn append `style_reminder` vào user message cuối.
- `ContextManager` đã trừ `style_reminder_tokens`, nhưng vẫn dùng heuristic `chars_per_token=3.0`.

### 2.3 Phần cứng

- CPU: Intel Pentium Silver **N5000**, 4 cores / 4 threads, ~1.1 GHz.
- RAM: **3.8 GB** tổng, chỉ ~0.4 GB free tại thời điểm kiểm tra.
- GPU: Intel UHD 605, không có dGPU/CUDA.
- Đĩa X: còn ~10.7 GB.
- Python runtime: 3.14.7 trong `.venv`; máy có thêm Python 3.13.
- Kết luận: **full fine-tune không khả thi để chạy thường xuyên trên máy này**; LoRA CPU chỉ nên chạy smoke test hoặc full run có kiểm soát/được duyệt.

### 2.4 Lỗi/rủi ro đã xác minh

1. **Ngân sách context bị đếm thiếu**:
   - System prompt trong `Modelfile`: 1.883 ký tự.
   - Với heuristic hiện tại `3 chars/token`: ~627 token.
   - `config.py` chỉ reserve **256 token**.
   - `style_reminder`: 431 ký tự ~143 token.
   - `message_budget` hiện tại = `2048 - 512 - 256 - 143 = 1137`.
   - Nếu tính đúng system prompt: `2048 - 512 - 627 - 143 = 766`.
   - Hệ quả: context có thể vượt `num_ctx` ~371 token, gây cắt/truncation hoặc giảm chất lượng.
2. `core/validator.py` còn `print("")` ở cuối file → in dòng trống khi import.
3. `run.bat` gọi `ollama create` mỗi lần chạy → tốn thời gian; `create.bat`/`run.bat` chưa tách rõ.
4. `starmentai.py` gọi API trực tiếp, trùng lặp logic; nên chuyển thành health-check.
5. `tests/benchmark_cpu.py` dùng `keep_alive=5m`, `num_ctx=1024`, threads 1/2/4; chưa đo cold/warm, RAM, median.
6. `data/starment_dataset.jsonl` có system prompt lặp lại trong từng mẫu, không đủ phân loại, không có eval.

## 3. Quyết định & giả định mặc định

- **Context**: ưu tiên tokenizer thật của Qwen3; nếu không cài được trên Python 3.14, dùng fallback conservative `chars_per_token=2.5` và reserve system prompt tối thiểu 800 token.
- **Model candidate**: tạo `starmentai-candidate`, giữ nguyên `starmentai` để rollback.
- **Dataset**: mục tiêu 500–800 mẫu thật/đã kiểm duyệt; tối thiểu 300 mẫu để smoke test. Nếu nguồn dữ liệu thật không đủ, giảm số lượng và ghi rõ, không bù bằng synthetic.
- **Fine-tune**: mặc định chỉ chạy **smoke test** LoRA trên CPU; full run là decision gate sau khi có số đo step-time/RAM.
- **System prompt**: freeze trước khi build dataset cuối; nếu đổi prompt, phải sinh lại dataset.
- **Eval**: tách `dev`/`test`; không dùng test để chọn tham số.

## 4. Phases triển khai

### Phase 0 — Freeze baseline

- Tạo branch `feature/starmentai-quality` (nếu user đồng ý commit; nếu không, dùng diff patch).
- Lưu baseline:
  - `reports/baseline_eval.json`
  - `reports/baseline_benchmark.json`
  - `git rev-parse HEAD`
- Backup dataset hiện tại: `data/archive/starment_dataset.initial.jsonl`.
- Không đổi model production trong phase này.

### Phase 1 — Sửa tính đúng đắn (P0)

**Mục tiêu**: context budget chính xác, import sạch, entrypoint rõ.

Files:

- `core/token_counter.py` (mới)
  - Optional backend `tokenizers.Tokenizer.from_file("data/qwen3_tokenizer.json")`.
  - Fallback heuristic; có `count_text()` và `count_messages()`.
  - Cache tokenizer; test được không cần network.
- `data/qwen3_tokenizer.json` (mới, nếu tải được)
  - Export từ `Qwen/Qwen3-0.6B` bằng `AutoTokenizer` trong env training.
- `config.py`
  - `system_prompt_tokens` không hardcode 256; tính conservative từ `system_prompt` + tokenizer/fallback.
  - Thêm `context_safety_margin` (đề xuất 64 token).
  - Rút gọn `style_reminder` xuống ~70–100 token; giữ đúng các rule quan trọng.
  - Cân nhắc thêm `presence_penalty`/`min_p` sau benchmark.
- `core/context_manager.py`
  - Nhận `token_counter` (hoặc classmethod factory).
  - `message_budget = num_ctx - reserved - system_prompt - style_reminder - safety_margin`.
  - Test trường hợp system prompt dài, style reminder dài, history nhiều.
- `core/validator.py`
  - Xóa `print("")`.
  - Thêm test redirect stdout để tránh tái phát.
- `main.py` / `starmentai.py`
  - `starmentai.py` chuyển thành health-check dùng `OllamaClient`.
  - `main.py` giữ nguyên logic chính; thêm log lỗi rõ nếu context/context_usage bất thường.
- `run.bat` / `create.bat`
  - `create.bat`: kiểm tra Ollama, tạo model.
  - `run.bat`: không recreate mỗi lần; chỉ tạo nếu model chưa tồn tại, sau đó `ollama run`.
  - Thêm `start.bat` hoặc hướng dẫn chạy `main.py` nếu cần.
- `tests/test_context.py`, `tests/test_ollama.py`, `tests/test_validator.py`
  - Test token counter, safety margin, style reminder budget, no stdout import.

**Acceptance Phase 1**:

- `pytest -q` pass.
- Với history dài, `estimated_tokens <= num_ctx - safety_margin`.
- System prompt + style reminder + reserved tokens không vượt `num_ctx`.
- Không còn `print("")` khi import.

### Phase 2 — Eval harness (P0)

**Mục tiêu**: có bằng chứng trước/sau, không đánh giá bằng cảm tính.

Files:

- `eval/eval_set.jsonl` (mới)
  - 120–160 prompt; chia `dev` 60–80, `test` 60–80.
  - Categories bắt buộc:
    - small talk/style (10%)
    - Vietnamese natural language (10%)
    - knowledge (15%)
    - coding/technical (15%)
    - multi-turn/coreference (15%)
    - ambiguous context (10%)
    - unknown/abstain (10%)
    - safety (10%)
    - length control (5%)
  - Mỗi item có `id`, `category`, `messages`, `reference`, `checks`, `rubric`.
- `eval/run_eval.py`
  - Gọi Ollama qua `OllamaClient`; lưu raw outputs.
  - Hỗ trợ `--model`, `--split`, `--think`, `--output`.
- `eval/score.py`
  - Deterministic: `contains`, `regex`, `json`, `length`.
  - Context recall: kiểm tra entity/key fact từ turn trước.
  - Abstain: có thừa nhận không biết, không khẳng định sai.
  - Language/style/safety: heuristic + cờ cho manual review.
- `eval/compare.py`
  - So sánh 2 report; output `reports/compare.md`.
- `tests/test_eval_set.py`
  - Validate schema, category coverage, duplicate IDs, no empty reference, split disjoint.
- `reports/baseline_eval.json`
  - Chạy trên `starmentai` hiện tại trước mọi thay đổi model.

**Metrics**:

- Correctness (exact/contains/regex theo loại).
- Context recall (multi-turn).
- Abstention rate trên unknown set.
- Hallucination flags.
- Style score (manual 1–5; hoặc rubric có người chấm).
- Safety violations.
- Latency: total, prompt eval, generation.
- Throughput: tokens/s.
- RAM: `ollama ps`/benchmark.

**Acceptance Phase 2**:

- Eval chạy được trên baseline và candidate.
- Có ít nhất 1 báo cáo baseline JSON + Markdown.
- Test eval set pass.

### Phase 3 — Prompt/context/parameter tuning (P1)

**Mục tiêu**: cải thiện nhanh, không cần train; tạo input ổn định cho dataset.

Tasks:

1. **A/B system prompt**
   - Bản A: hiện tại (1.883 ký tự).
   - Bản B: rút gọn ~900–1.200 ký tự, giữ persona + safety + language + abstain + length rules.
   - Đo trên eval dev: style, correctness, token count, latency.
2. **A/B style reminder**
   - A: 431 ký tự hiện tại.
   - B: 70–100 token, chỉ nhắc 3–4 rule dễ vi phạm.
   - C: bỏ hẳn (nếu eval không giảm chất lượng).
3. **Parameter sweep**
   - `num_ctx`: 2048 vs 3072 (RAM permitting).
   - `num_thread`: 1 vs 2 vs 4.
   - `num_predict`: 256 vs 512.
   - `temperature`: 0.5/0.6/0.7; `top_p`: 0.8/0.9; `repeat_penalty`: 1.0/1.1.
   - `keep_alive`: 30s vs 5m.
   - `think`: adaptive vs always false vs always true (đo trên subset 30 prompt).
4. **Context manager**
   - Dùng token counter; thêm safety margin.
   - Test hội thoại 10+ lượt; đảm bảo latest user message luôn được giữ.
   - Nếu recall < baseline, cân nhắc pin facts đơn giản (tên/sở thích/ràng buộc) thay vì tăng `num_ctx`.
5. **Benchmark**
   - Nâng cấp `tests/benchmark_cpu.py`:
     - `--threads 1,2 --ctx 2048 --keep-alive 30s --runs 3`.
     - Đo cold/warm, median, prompt tokens, eval tokens, tokens/s.
     - Ghi JSON.

**Acceptance Phase 3**:

- Chọn được bộ prompt/params có eval dev không giảm correctness/context recall.
- `pytest -q` pass.
- Benchmark có JSON; không có regression latency/RAM quá ngưỡng.

### Phase 4 — Dataset (P1)

**Mục tiêu**: 500–800 mẫu thật, sạch, đúng schema, không synthetic filler.

#### 4.1 Nguồn dữ liệu

Ưu tiên:

- `vlinhd11/vietnamese-sft-10k` — CC BY 4.0; phải lọc 30% noisy/toxic.
- `dungnvt/vi_instruct_general_dataset_cleaned` — ShareGPT, có category; **phải xác minh license** trước khi dùng.
- `aisingapore/MultiTurn-Chat-MT-Bench` — CC BY 4.0; chỉ dùng cho eval, không train.
- Dữ liệu do user/owner duyệt: chat thật, câu hỏi kỹ thuật, code đã test.
- Coding: dùng ví dụ code thật đã chạy/đã kiểm tra; không bịa API/lệnh.

Tránh:

- `DataOpsFusion/lunet-vi-chat` (synthetic, proprietary).
- `luanngo/Vietnamese-Legal-Chat-Dataset` (non-commercial).
- Dataset chưa rõ license hoặc machine-translation chất lượng thấp.

#### 4.2 Schema

```json
{
  "id": "train-000001",
  "source": "vlinhd11/vietnamese-sft-10k",
  "license": "CC-BY-4.0",
  "category": "coding",
  "messages": [
    {"role": "system", "content": "<SYSTEM_PROMPT_FROZEN>"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tags": ["python", "debug"],
  "split": "train"
}
```

#### 4.3 Pipeline

Files:

- `data/scripts/fetch_sources.py` — tải dataset đã duyệt; ghi `data/raw/source_manifest.json`.
- `data/scripts/clean_dataset.py` — chuẩn hóa, lọc, dedupe.
- `data/scripts/check_dataset.py` — quality gate.
- `data/scripts/split_dataset.py` — train/val/test theo source/time, chống leakage.
- `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`
- `data/processed/dataset_stats.json`
- `data/dataset_card.md` — nguồn, license, số lượng, giới hạn.

Quality gates:

- Bỏ empty/duplicate exact; near-duplicate bằng MinHash/SimHash.
- Độ dài 40–800 token; bỏ quá ngắn/quá dài.
- Language check: tiếng Việt chiếm tỷ lệ hợp lý; cho phép code block.
- PII/secret regex: số điện thoại, email, token, password.
- Safety/toxicity: loại mẫu vi phạm; giữ mẫu khó để test.
- Factual spot-check 10% + toàn bộ coding samples phải chạy/test được.
- Train/val/test không trùng `id`, không overlap n-gram cao.
- Không thêm mẫu synthetic; nếu thiếu category, ghi `gap` trong dataset card.

#### 4.4 Phân bố mục tiêu

- 500–800 mẫu train; 100–150 val; 100 test.
- Tối thiểu 300 mẫu nếu chỉ smoke test.
- Categories: small talk/style, Vietnamese, knowledge, coding, multi-turn/coreference, ambiguous, abstain, safety.
- 5–10% mẫu "không biết"/"cần thêm thông tin".

**Acceptance Phase 4**:

- `python data/scripts/check_dataset.py` pass.
- `pytest -q` pass.
- Dataset card ghi rõ nguồn/license/số lượng.
- Không có synthetic filler.

### Phase 5 — LoRA pipeline + smoke test (P1)

**Mục tiêu**: có pipeline huấn luyện tái lập được, chạy được smoke test trên máy hiện tại; không hứa full run.

Files:

- `training/requirements-training.txt`
  - `torch` CPU, `transformers>=4.51`, `peft`, `datasets`, `accelerate`, `safetensors`.
- `training/train_lora.py`
  - Load `Qwen/Qwen3-0.6B`.
  - LoRA: `r=8`, `alpha=16`, `dropout=0.05`, target `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
  - `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, `max_seq_len=512`, `num_train_epochs=3`, `lr=1e-4`, `gradient_checkpointing=True`, `use_cpu=True`, `bf16=False`, `fp16=False`.
  - Mask prompt tokens (`labels=-100`), chỉ train assistant response.
  - `--max_steps` cho smoke test; `--dry-run` kiểm tra dataset/tokenize.
  - Lưu `adapter_config.json`, `adapter_model.safetensors`, logs.
- `training/merge_lora.py`
  - Merge adapter vào base; lưu `training/artifacts/merged/`.
- `training/export_ollama.py` hoặc `Modelfile.candidate`
  - Option A: `ADAPTER` trong Modelfile (nếu Ollama nhận adapter tương thích).
  - Option B: convert GGUF + `Modelfile.candidate` `FROM ./merged.gguf`.
- `training/README.md` — lệnh chạy, RAM/disk, giới hạn.
- `training/train.bat` — full run có xác nhận.

Smoke test:

```powershell
py -3.13 -m venv .venv-training
.venv-training\Scripts\python -m pip install -r training/requirements-training.txt
.venv-training\Scripts\python training\train_lora.py --data data\processed\train.jsonl --max_steps 5 --output training\artifacts\smoke
```

**Acceptance Phase 5**:

- Smoke test hoàn tất, adapter được lưu, loss giảm, không OOM.
- Ghi `reports/smoke_train.json`: step time, RAM, disk.
- Có lệnh full run nhưng **không tự chạy** nếu chưa duyệt.

### Phase 6 — Full training (optional, decision gate)

Chỉ chạy nếu user duyệt và có:

- RAM free > 3 GB khi train (đóng app khác).
- Disk trống > 10 GB.
- Smoke test cho step-time hợp lý.
- Dataset đã pass quality gate.

Kế hoạch:

- 300–500 mẫu, 3 epochs, batch 1, grad accum 8, max_seq 512.
- Chạy nền, log ra `training/logs/`.
- Không chạy khi user đang dùng máy cho việc khác.
- Nếu OOM: giảm `max_seq_len` 384/256, tăng grad accum, hoặc chuyển sang Colab/GPU.

### Phase 7 — Integration & A/B (P0)

- Build `starmentai-candidate` từ `Modelfile.candidate`.
- Chạy eval `dev` + `test` trên `starmentai` vs `starmentai-candidate`.
- Chạy benchmark cold/warm.
- Chỉ promote candidate nếu:
  - Correctness/context recall không giảm quá 5%.
  - Style score >= 4/5.
  - Abstention >= 80% trên unknown set.
  - Không có safety violation.
  - p95 latency không tăng quá 20%.
  - RAM không vượt 4.0 GB.
- Nếu candidate tệ hơn: giữ `starmentai`, ghi report, rollback.

Rollback:

- `ollama rm starmentai-candidate` (nếu cần).
- `git checkout` branch/commit baseline.
- Giữ `Modelfile` baseline trong git history.

### Phase 8 — Final validation & docs

- `pytest -q` full pass.
- `python eval/run_eval.py --model starmentai --output reports/final.json`.
- `python tests/benchmark_cpu.py ... > reports/final_benchmark.json`.
- Cập nhật `README.md`: cách chạy, train, eval, rollback.
- Báo cáo cuối: trước/sau, bằng chứng, giới hạn, dataset card.

## 5. Acceptance criteria tổng

| Hạng mục | Target |
|---|---|
| Unit test | 100% pass |
| Context overflow | 0; `estimated_tokens <= num_ctx - safety_margin` |
| Correctness eval | Không giảm >5% so với baseline |
| Context recall | Tăng hoặc không giảm |
| Abstention | >=80% trên unknown set |
| Safety | 0 violation nghiêm trọng |
| Style | >=4/5 (manual rubric) |
| Latency | p95 không tăng >20% |
| Throughput | Không giảm >10% |
| RAM | <=4.0 GB khi chạy |
| Dataset | >=300 mẫu thật; 500–800 nếu đủ nguồn; không synthetic filler |
| Reproducibility | Smoke test + lệnh full run + rollback |

## 6. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| N5000 + 3.8GB RAM không train nổi | Smoke test trước; fallback prompt/context; Colab/GPU nếu user duyệt |
| Python 3.14 thiếu wheel | Dùng Python 3.13 cho training env; runtime giữ 3.14 |
| Tokenizer không cài được | Fallback 2.5 chars/token + reserve 800 system tokens |
| Dataset license/noise | Verify license; lọc toxicity; dataset card; loại synthetic |
| LoRA làm giảm knowledge | A/B eval; giữ baseline; không promote nếu regression |
| Context budget sai | Token counter + safety margin + test |
| Ollama adapter không tương thích | Merge + GGUF export; test `ollama create` |
| Full run quá lâu | Decision gate; log step-time; dừng nếu không hợp lý |

## 7. Ngoài phạm vi

- Pretraining lại từ đầu.
- RAG/vector DB (chỉ note nếu recall còn kém).
- Web UI/API server.
- Fine-tune 1.7B/7B.
- Tự động crawl dữ liệu không xin phép.
- Dùng dataset synthetic để tăng số lượng.

## 8. Thứ tự thực thi đề xuất

1. Phase 0 + Phase 1 (song song với chuẩn bị eval set).
2. Phase 2 baseline eval.
3. Phase 3 prompt/context/params.
4. Phase 4 dataset (freeze system prompt trước).
5. Phase 5 LoRA smoke test.
6. Phase 6 full run nếu được duyệt.
7. Phase 7 A/B + Phase 8 final.

## 9. Câu hỏi/quyết định còn mở (mặc định nếu không phản hồi)

1. **Full fine-tune**: mặc định **không chạy**; chỉ smoke test. Nếu muốn full run, cần duyệt và chấp nhận thời gian/RAM.
2. **Nguồn dataset**: mặc định dùng dataset CC BY/permissive đã verify + dữ liệu owner duyệt; không dùng synthetic.
3. **Model candidate**: mặc định `starmentai-candidate`; không ghi đè `starmentai`.
4. **Tokenizer**: mặc định thử `tokenizers` + `qwen3_tokenizer.json`; fallback heuristic nếu không có wheel.