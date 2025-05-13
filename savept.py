import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Đường dẫn tới mô hình đã lưu bằng trainer.save_model()
# Giữ nguyên raw string hoặc đổi sang dấu gạch chéo thuận cho đường dẫn trên Windows
model_dir = r"C:\Users\Your mom\Downloads\mddial_t5_base_finetuned_diagnosis_f16false\checkpoint-6300"

# Load lại mô hình (chỉ phần model)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# KHÔNG LƯU TRỰC TIẾP ĐỐI TƯỢNG TOKENIZER.
# Thay vào đó, chỉ lưu tên của tokenizer (hoặc đường dẫn tới nó nếu bạn muốn tải lại cục bộ).
# Ở đây, bạn đang tải từ "t5-base", nên hãy lưu tên này.

# Tạo dictionary chứa chỉ trọng số mô hình và thông tin cần thiết để tải lại tokenizer
oracle = {
    "model_state_dict": model.state_dict(),  # Đổi key từ "model" thành "model_state_dict" để rõ ràng hơn
    "model_class": "T5ForConditionalGeneration",
    "tokenizer_name_or_path": "google/flan-t5-base", # LƯU TÊN CỦA TOKENIZER
    "tokenizer_class": "T5Tokenizer", # Lưu class tokenizer để dễ load lại
}

# Lưu thành một file .pt duy nhất
torch.save(oracle, "mddial_t5_base_oracle.pt")
print("✅ Đã lưu mô hình vào mddial_t5_base_oracle.pt")