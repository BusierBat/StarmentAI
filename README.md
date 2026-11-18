# StarmentAI 🤖

**StarmentAI** là dự án trợ lý trí tuệ nhân tạo chạy cục bộ trên máy tính, được phát triển bằng Python và sử dụng Ollama để vận hành mô hình AI.

Dự án được tạo ra với mục tiêu nghiên cứu, học hỏi, thử nghiệm công nghệ AI và chia sẻ kiến thức lập trình.

## 👥 Người thực hiện

Dự án được phát triển bởi ba thằng rảnh lồn đéo có gì làm:

**1. @datlgbtgaylo**

TikTok: https://www.tiktok.com/@datlgbtgaylo

**2. @leg_bao110**

TikTok: https://www.tiktok.com/@leg_bao110

**3. @orrrrrzzzzzz1**

TikTok: https://www.tiktok.com/@orrrrrzzzzzz1

Thay vì ngồi hàng giờ lướt TikTok vô nghĩa, ba thành viên quyết định bắt tay xây dựng một dự án AI của riêng mình.

Mỗi người đóng góp ý tưởng, nghiên cứu và phát triển dự án theo khả năng của mình. Không có quy trình quản lý phức tạp, thích nghiên cứu gì thì cùng nhau nghiên cứu, thích phát triển gì thì bắt tay vào làm.

Dự án có sự hỗ trợ của các công cụ AI trong quá trình nghiên cứu và phát triển.

## 🚀 Giới thiệu dự án

StarmentAI là dự án AI mã nguồn mở được viết bằng Python.

Nguồn gốc dự án khá đặc biệt: ban đầu, hai thành viên rủ nhau xóa System32 vì quá chán và không có gì làm. Sau đó, thay vì tiếp tục những trò nghịch ngợm vô nghĩa, cả hai quyết định hợp tác xây dựng một trợ lý AI của riêng mình.

Thành viên thứ ba sau đó gia nhập, đưa đội ngũ phát triển lên con số ba.

Từ một ý tưởng ngẫu hứng, StarmentAI dần trở thành một dự án thực tế để cả nhóm học hỏi và thử thách bản thân.

### Công nghệ sử dụng

- **Ngôn ngữ lập trình:** Python.

- **AI Runtime:** Ollama.

- **Mô hình nền:** Qwen3 0.6B.

- **Quản lý mã nguồn:** Git và GitHub.

Model được sử dụng: [Qwen3 0.6B trên Ollama Library](https://ollama.com/library/qwen3).

## ⚙️ Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt Ollama

Trước tiên, cần cài đặt Ollama để chạy mô hình AI trên máy tính.

Tải Ollama tại:

https://ollama.com/download

Chọn phiên bản phù hợp với hệ điều hành đang sử dụng.

### 2. Tải mô hình Qwen3 0.6B

Sau khi cài đặt Ollama, mở Terminal hoặc PowerShell và chạy:

```powershell
ollama pull qwen3:0.6b
```

Đợi quá trình tải model hoàn tất.

Có thể kiểm tra model bằng lệnh:

```powershell
ollama list
```

### 3. Tải mã nguồn StarmentAI

Truy cập repository chính thức:

https://github.com/BusierBat/StarmentAI_

Tải source code về máy hoặc clone repository bằng Git:

```powershell
git clone https://github.com/BusierBat/StarmentAI_.git
```

Sau đó mở thư mục dự án.

### 4. Khởi tạo StarmentAI

Chạy file:

```text
create.bat
```

File này được sử dụng để tạo model StarmentAI từ cấu hình dự án.

Chờ quá trình khởi tạo hoàn tất.

### 5. Khởi động StarmentAI

Chạy file:

```text
run.bat
```

hoặc trực tiếp:

```powershell
python main.py
```

StarmentAI sẽ khởi động với giao diện TUI dạng terminal giống opencode:

- Header bar: tên app, model, dung lượng context, version.
- Vùng chat: message của bạn và StarmentAI phân biệt màu, phản hồi được render markdown.
- Hộp input có viền phía dưới, streaming từng token khi model trả lời.
- Footer gợi ý phím tắt.

**Phím tắt:**

- `Enter`: gửi tin nhắn
- `Ctrl+Enter` / `Ctrl+J`: xuống dòng trong input
- `↑` / `↓`: duyệt lịch sử tin nhắn (khi input một dòng)
- `Ctrl+C` / `Ctrl+Q`: thoát

**Lệnh trong chat:**

- `/help`: hiển thị các lệnh
- `/clear`: xóa lịch sử hội thoại
- `/exit`: thoát

**Chế độ khác:**

```powershell
python main.py --demo     # Chạy TUI với processor giả, không cần Ollama
python main.py --plain    # Chạy giao diện terminal đơn giản như cũ
```

**Lưu ý:**

- Ollama cần được cài đặt và hoạt động.

- Model `qwen3:0.6b` cần được tải trước.

- Đảm bảo đã thực hiện đầy đủ các bước khởi tạo.

- Nếu gặp lỗi, hãy kiểm tra thông báo trong Terminal hoặc tạo Issue trên GitHub.

## 🎯 Mục tiêu dài hạn

StarmentAI hiện đang được phát triển bằng Python.

Sau khi hoàn thiện phiên bản Python và tích lũy đủ kiến thức về Rust, cả nhóm dự định:

1. Nghiên cứu sâu hơn về ngôn ngữ Rust.

2. Tìm hiểu cách xây dựng và tối ưu ứng dụng bằng Rust.

3. Thiết kế lại kiến trúc StarmentAI phù hợp với Rust.

4. Rewrite dự án sang Rust.

5. Build thành file `.exe` để thuận tiện triển khai trên Windows.

Mục tiêu không chỉ là chuyển đổi ngôn ngữ lập trình, mà còn cải thiện hiệu năng, khả năng quản lý tài nguyên và độ ổn định của ứng dụng.

## 🤝 Đóng góp

StarmentAI là dự án mã nguồn mở.

Mọi người có thể:

- Tham khảo mã nguồn.

- Đưa ra ý tưởng cải tiến.

- Báo cáo lỗi.

- Đề xuất tính năng mới.

- Đóng góp mã nguồn thông qua Pull Request.

Nếu phát hiện lỗi, hãy tạo Issue trên GitHub để nhóm có thể xem xét và xử lý.

Repository chính thức:

https://github.com/BusierBat/StarmentAI_

---

## 📌 Lời kết

StarmentAI được xây dựng từ sự tò mò, tinh thần học hỏi và một chút... bất ổn của ba thành viên.

Không phải một công ty công nghệ lớn, không có ngân sách khổng lồ, cũng chẳng có đội ngũ kỹ sư hàng trăm người.

Chỉ có ba thằng cùng nhau học hỏi, thử nghiệm và phát triển một dự án AI của riêng mình.

**===== NẾU CÓ LỖI GÌ THÌ KỆ MẸ TỤI BÂY =====**

**===== HẾT =====**