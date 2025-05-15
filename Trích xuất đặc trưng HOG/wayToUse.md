<h1>🧠💡 HOG - Resnet mạnh hơn tao 😎</h1>

<h2>🦾 HOG là gì?</h2>

Là Histogram of Oriented Gradients – nghe oách chứ code 1 dòng là xong 😆

    hog_features = np.array([hog(rgb2gray(img),pixels_per_cell=(16, 16), cells_per_block=(1, 1),visualize=False) for img in images])

<h2>🔍 Cách học siêu nhanh</h2>

<h3>✅ Cách 1:</h3>
Xem ảnh ví dụ theo thứ tự: 50 ➡️ 57 ➡️ 09 ➡️ 11

(ảnh là best teacher mà 😘)

<h3>✅ Cách 2:</h3>

Đọc phần “văn tắt học dài lâu” dưới đây – đảm bảo vừa học vừa cười 🤭

<h4>🧩 Trước khi vô, "giải mã" mấy từ chuyên môn :</h4>

    😏 pixel là điểm ảnh, thành phần nhỏ nhất để tạo lên 1 bức ảnh 
    
    😏 cell là nhiều điểm ảnh gần nhau, tạo thành 1 ma trận điểm ảnh 
    
    😏 block là nhiều cell, tạo thành 1 ma trận cell 
    
    📉 Gradient: Chỉ hướng thay đổi của ảnh, tưởng tượng như gió thổi á 🎐
    
    🎯 Bin: 9 hướng chia đều từ 0° đến 180° → mỗi hướng chiếm 20° → đúng chuẩn 9 bin.

<h4>🎬 5 ý nắm trọn HOG </h4>

1️⃣ Xử lý ảnh (Chỉnh kích thước, màu sắc) 👩‍🍳
- Input: ảnh 
- Output: 1 ma trận pixel

Resize ảnh (vd: 112x112 hoặc 224x224)

Chuyển về đen trắng cho "gọn nhẹ dễ xử" 🖤

2️⃣ Tính Gradient – Đạo hàm của ảnh 🌀
- Input: 1 ma trận pixel 
- Output: 2 ma trận cùng kích cỡ (1 là ma trận hướng H, 1 là ma trạn cường độ C)

Mỗi pixel → tính độ lớn (mạnh yếu) và hướng (trái phải, trên dưới…)

3️⃣ Tính vecto đặc trưng cho từng cell 🧮
- Input: 2 ma trận cùng kích cỡ (1 là ma trận hướng H, 1 là ma trạn cường độ C), kích thước của mỗi ma trận cell 
- Output: các vecto kích thước 1x9 (mỗi cell có 1 vecto riêng)

Mỗi cell (16x16) → lấy 9 hướng → ra vector 1x9 → mỗi cell có 9 đặc trưng  

Ảnh 112x112 → có 49 cell → 49 vecto

(112/16=7, 7x7=49)

4️⃣ Chuẩn hóa theo block 💪
- Input: các vecto tương ứng với cell, kích thước block 
- Output: các vecto kích thước 1x[9.(số cell/block)] (mỗi block có 1 vecto riêng)

1 block = 2x2 cell → mỗi block có 4x9 = 36 đặc trưng

Có 36 block → 36 vector 1x36

(7-2+1)^2 = 36 🧠

5️⃣ Tính vecto đặc trưng HOG 

Vector HOG cuối cùng:

📏 Số block được duyệt × Số cell/block × 9 bin

= 36 × 4 × 9 = 1296

➡️ Vậy được vector 1x1296 → 1 ảnh có 1296 đặc trưng → Hơi kém :<<<💥

