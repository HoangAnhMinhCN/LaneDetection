# Lane Detection Studio

Nhận diện làn đường từ ảnh hoặc video bằng OpenCV, kèm giao diện Gradio.
Tải một clip lên để xem video đã đánh dấu làn đường, đo tốc độ xử lý và tải
thống kê theo từng frame. Không cần GPU.

## Cài đặt lần đầu

Yêu cầu Git và Python 3.10 trở lên (đã chạy thử với Python 3.11).
FFmpeg được cung cấp bởi `imageio-ffmpeg`, không cần cài riêng trên Windows.

```sh
git clone https://github.com/HoangAnhMinhCN/LaneDetection.git
cd LaneDetection
```

Nếu đã tải repo, mở terminal ngay trong thư mục dự án và bỏ qua bước clone.

### Windows — PowerShell hoặc CMD

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu máy không nhận lệnh `python`, dùng `py -3 -m venv .venv` ở bước tạo môi trường.

### Linux / macOS

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Các cách mở giao diện

Chỉ cần chọn **một** cách bên dưới. Giữ terminal chạy trong lúc sử dụng;
đóng terminal hoặc bấm `Ctrl+C` sẽ dừng ứng dụng.

### Cách 1 — Chạy trực tiếp, không cần kích hoạt môi trường

**Windows (PowerShell hoặc CMD):**

```powershell
.\.venv\Scripts\python.exe app.py
```

**Linux / macOS:**

```sh
./.venv/bin/python app.py
```

Sau khi server khởi động, mở một trong hai địa chỉ sau bằng trình duyệt
trên cùng máy:

- [http://127.0.0.1:7860](http://127.0.0.1:7860)
- [http://localhost:7860](http://localhost:7860)

Ứng dụng không tự bật trình duyệt; có thể bấm link trong terminal hoặc dán
địa chỉ vào Chrome, Edge hay Firefox.

### Cách 2 — Kích hoạt môi trường rồi chạy

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

**Windows CMD:**

```bat
.venv\Scripts\activate.bat
python app.py
```

**Linux / macOS:**

```sh
source .venv/bin/activate
python app.py
```

Mở [http://127.0.0.1:7860](http://127.0.0.1:7860).
Nếu PowerShell chặn `Activate.ps1`, dùng **Cách 1** để chạy trực tiếp.

### Cách 3 — Mở từ VS Code

1. Chọn **File → Open Folder** và mở thư mục repo.
2. Cài extension **Python** của Microsoft nếu chưa có.
3. Bấm `Ctrl+Shift+P` → **Python: Select Interpreter** → chọn Python trong `.venv`.
4. Mở [app.py](./app.py), chọn **Run Python File in Terminal** ở góc trên bên phải.
5. Bấm địa chỉ `http://127.0.0.1:7860` xuất hiện trong terminal để mở giao diện.

Cũng có thể mở **Terminal → New Terminal** và dùng lệnh trong **Cách 1**.

## Sử dụng giao diện

1. Tải clip vào **Clip đầu vào**, hoặc chọn một clip mẫu từ [Input/](./Input/).
2. Nếu cần, mở **Tùy chỉnh nhận diện** để chỉnh chiều rộng xử lý, độ làm mượt
   và độ dài vạch hiển thị. Giảm chiều rộng để chạy nhanh hơn; đầu ra giữ độ
   phân giải gốc.
3. Bấm **Chạy nhận diện** và chờ thanh tiến trình hoàn tất.
4. Xem video ở **Video đã nhận diện**, bấm nút tải xuống để lưu MP4.
5. Xem bảng thống kê và tải CSV từng frame hoặc JSON tổng hợp.

Video đầu ra dùng H.264 để phát trên trình duyệt và giữ âm thanh nếu clip gốc có.
Pipeline nhận diện gồm mặt nạ màu trắng/vàng, Gaussian blur, Canny, vùng quan tâm,
Hough transform, ước lượng hai vạch và làm mượt theo thời gian.

## Các thống kê đo được

| Chỉ số | Ý nghĩa |
| --- | --- |
| Độ phân giải, FPS nguồn, thời lượng | Thông tin video và thời lượng đã xử lý |
| Số frame đã xử lý | Số frame đọc và ghi trong lượt chạy |
| FPS pipeline | Tốc độ đọc, nhận diện và ghi frame; chưa gồm xuất H.264 |
| Thời gian detector / frame | Thời gian trung bình chỉ dành cho nhận diện |
| Tổng thời gian | Thời gian xử lý, gồm cả mã hóa video H.264 |
| Frame tìm mới vạch trái/phải/cả hai | Số lượng và tỷ lệ frame có đoạn vạch được chấp nhận sau lọc nhiễu |
| Frame hiển thị đủ hai làn | Số lượng và tỷ lệ frame có cả hai ước lượng làn để vẽ |

**Lưu ý:** "Tìm mới" chỉ tính bằng chứng trong frame hiện tại. "Hiển thị" có thể
dùng ước lượng giữ từ frame trước để làm mượt. Các tỷ lệ này đo độ phủ phát hiện,
không phải accuracy/confidence vì chưa có nhãn ground truth.

Mỗi lượt chạy tạo một thư mục riêng trong `runs/` gồm:

- `detected.mp4`: video kết quả.
- `frames.csv`: thời điểm, thời gian nhận diện và trạng thái phát hiện/hiển thị từng frame.
- `summary.json`: thống kê tổng hợp và tham số nhận diện.

Kết quả và môi trường `.venv` được bỏ qua bởi Git. Clip dài có thể cần nhiều thời
gian và dung lượng; có thể xóa các thư mục kết quả trong `runs/` khi không cần nữa.

## Chạy bằng dòng lệnh, không mở giao diện

Sau khi cài thư viện, dùng Python của môi trường `.venv`:

```powershell
# Windows: xử lý video
.\.venv\Scripts\python.exe lane_detection.py Input/input.mp4 result.mp4

# Windows: vừa xử lý vừa xem cửa sổ OpenCV (bấm q để dừng sớm)
.\.venv\Scripts\python.exe lane_detection.py Input/input.mp4 result.mp4 --show

# Windows: xử lý ảnh
.\.venv\Scripts\python.exe lane_detection.py road.jpg result.jpg
```

Trên Linux/macOS, thay `.\.venv\Scripts\python.exe` bằng `./.venv/bin/python`.
Lệnh CLI gốc chỉ xuất ảnh/video; dùng giao diện Gradio để có thêm CSV/JSON
và xuất MP4 H.264 kèm âm thanh.

## Xử lý lỗi thường gặp

- **Không truy cập được giao diện:** chạy ứng dụng trước, đợi server khởi động và
  giữ terminal mở. Địa chỉ mặc định chỉ dùng trên máy đang chạy ứng dụng.
- **Cổng 7860 đang được sử dụng:** dừng phiên ứng dụng cũ bằng `Ctrl+C`, hoặc đổi
  `server_port=7860` ở cuối [app.py](./app.py) sang cổng khác rồi mở đúng địa chỉ đó.
- **Thiếu thư viện:** chạy lại lệnh cài từ [requirements.txt](./requirements.txt)
  bằng Python của `.venv`, không dùng Python của môi trường khác.
- **Video không mở được:** thử MP4 với codec H.264 và kiểm tra clip phát được trên máy.
- **Xử lý chậm:** giảm chiều rộng xử lý trong giao diện hoặc thử clip ngắn hơn.
