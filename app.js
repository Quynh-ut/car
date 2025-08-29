// Upload ảnh test (jpg/png) + gửi API
document.getElementById("upload-btn").addEventListener("change", async (event) => {
  let file = event.target.files[0];
  if (!file) return;

  // Reset các kết quả trước đó
  document.getElementById("plate-number").textContent = "---";
  document.getElementById("confirmation").textContent = "⏳ Đang nhận diện...";
  document.getElementById("debug-info").innerHTML = "";
  document.getElementById("additional-plates").innerHTML = "";
  document.getElementById("image-preview").innerHTML = "";

  // Hiển thị ảnh được chọn
  const photoUrl = URL.createObjectURL(file);
  photo.src = photoUrl;
  
  // Hiển thị tên file
  document.getElementById("file-name").textContent = file.name;

  try {
    let formData = new FormData();
    formData.append("file", file);

    // Hiển thị trạng thái đang xử lý
    document.getElementById("confirmation").textContent = "⏳ Đang gửi ảnh và nhận diện...";

    let response = await fetch("http://127.0.0.1:8000/api/image/upload/", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
    }

    let result = await response.json();
    console.log("Kết quả nhận diện từ server:", result);

    // Hiển thị thông tin debug
    let debugHTML = `
      <h4>Thông tin xử lý:</h4>
      <p>Trạng thái: ${result.status}</p>
      <p>Số biển số tìm thấy: ${result.count}</p>
    `;

    if (result.original_image) {
      debugHTML += `<p>Ảnh gốc: <a href="http://127.0.0.1:8000/${result.original_image}" target="_blank">Xem</a></p>`;
    }

    document.getElementById("debug-info").innerHTML = debugHTML;

    if (result.plates && result.plates.length > 0) {
      // Hiển thị biển số thực tế nhận diện được
      document.getElementById("plate-number").textContent = result.plates[0];
      document.getElementById("confirmation").textContent = "✅ Nhận diện thành công";
      
      // Hiển thị tất cả biển số nếu có nhiều
      if (result.plates.length > 1) {
        document.getElementById("additional-plates").innerHTML = `
          <div style="margin-top: 10px; padding: 10px; background: #f0f8ff; border-radius: 5px;">
            <strong>Các biển số khác nhận diện được:</strong>
            <ul>
              ${result.plates.slice(1).map(plate => `<li>${plate}</li>`).join('')}
            </ul>
          </div>
        `;
      }
    } else {
      document.getElementById("plate-number").textContent = "---";
      document.getElementById("confirmation").textContent = "❌ Không nhận diện được biển số";
      document.getElementById("debug-info").innerHTML += `
        <p style="color: red;">Không tìm thấy biển số nào phù hợp trong ảnh</p>
        <p>Hãy thử với ảnh chụp rõ nét hơn hoặc điều chỉnh góc chụp</p>
      `;
    }
  } catch (err) {
    console.error("Lỗi khi upload ảnh:", err);
    document.getElementById("plate-number").textContent = "---";
    document.getElementById("confirmation").textContent = "⚠️ Lỗi kết nối API";
    document.getElementById("debug-info").innerHTML = `
      <p style="color: red;">Chi tiết lỗi: ${err.message}</p>
      <p>Kiểm tra xem server có đang chạy không: http://127.0.0.1:8000</p>
    `;
  }
});

// Làm mới
document.getElementById("refresh-btn").addEventListener("click", () => {
  photo.src = "";
  document.getElementById("plate-number").textContent = "---";
  document.getElementById("confirmation").textContent = "ĐANG CHỜ NHẬN DIỆN";
  document.getElementById("upload-btn").value = "";
  document.getElementById("debug-info").innerHTML = "";
  document.getElementById("additional-plates").innerHTML = "";
  document.getElementById("file-name").textContent = "";
});