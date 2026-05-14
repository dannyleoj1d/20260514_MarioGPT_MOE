import torch
import sys

def check_environment():
    print(f"--- 系統環境檢查 ---")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 檢查是否為 2.7+ (針對 Blackwell 優化的分水嶺)
    major, minor = map(int, torch.__version__.split('.')[:2])
    if major > 2 or (major == 2 and minor >= 7):
        print("✅ PyTorch 版本符合 2.7+ 建議規範。")
    else:
        print("⚠️ 警告: PyTorch 版本低於 2.7。舊版本可能無法調用 50 系列的 FP4/FP8 加速。")

    print(f"\n--- GPU 加速檢查 ---")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        capability = torch.cuda.get_device_capability(device_id)
        cuda_ver = torch.version.cuda
        
        print(f"使用裝置: {device_name}")
        print(f"CUDA 編譯版本: {cuda_ver}")
        print(f"計算能力 (Compute Capability): {capability}")

        # 測試 BF16 支援 (50 系列的強項)
        bf16_support = torch.cuda.is_bf16_supported()
        print(f"硬體是否支援 BF16: {'是' if bf16_support else '否'}")
        
        # 測試 Tensor Core 矩陣運算
        try:
            a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
            b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
            c = torch.matmul(a, b)
            print("✅ Tensor Core 矩陣運算測試成功。")
        except Exception as e:
            print(f"❌ 運算測試失敗: {e}")
    else:
        print("❌ 找不到 CUDA 裝置，請確認驅動程式已安裝。")

if __name__ == "__main__":
    check_environment()