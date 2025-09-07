import torch
import time
from TSMAE import TSMAE  # sửa path nếu TSMAE.py nằm nơi khác

def benchmark_tsmae():
    # --- Config ---
    input_file = "normal_sample0_column.txt"   # file đã chuyển thành cột
    checkpoint_file = "tsmae_weights.pth"

    # --- Khởi tạo model ---
    input_size = 1
    hidden_size = 10
    memory_size = 20
    sparsity_factor = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TSMAE(input_size, hidden_size, memory_size, sparsity_factor)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.to(device).eval()

    # --- Đọc input và reshape thành [1, seq_len, 1] ---
    with open(input_file, "r") as f:
        data = [float(line.strip()) for line in f if line.strip()]
    input_tensor = (
        torch.tensor(data, dtype=torch.float32)
             .unsqueeze(-1)  # [seq_len, 1]
             .unsqueeze(0)   # [1, seq_len, 1]
             .to(device)
    )

    # --- Đo thời gian và GPU memory ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"⏱ Runtime: {elapsed_time_ms:.3f} ms")

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"💾 Peak GPU memory: {peak_memory_mb:.2f} MB")

    # --- In output (hỗ trợ tuple) ---
    if isinstance(output, tuple):
        for idx, out in enumerate(output):
            print(f"⏺ Output[{idx}] shape: {tuple(out.shape)}")
            print(f"⏺ Output[{idx}]:\n{out.cpu().numpy()}\n")
    else:
        print(f"⏺ Output shape: {tuple(output.shape)}")
        print(f"⏺ Output:\n{output.cpu().numpy()}")

if __name__ == "__main__":
    benchmark_tsmae()
