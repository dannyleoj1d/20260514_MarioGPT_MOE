import os
import torch
from transformers import AutoTokenizer
from mario_gpt.dataset import MarioDataset

# ================= 設定區 =================
OUTPUT_FILE = "debug_continuous_view.txt" # 輸出檔案名稱
CHECK_COUNT = 300                          # 你想看連續的幾張圖？
CONTEXT_LEN = 700                         # 必須跟你訓練時設定的一樣
HEIGHT = 14                               # 瑪利歐關卡高度
# =========================================

def main():
    print("🚀 初始化 Tokenizer 與 Dataset...")
    
    # 1. 載入 Tokenizer (跟訓練時一樣)
    tokenizer = AutoTokenizer.from_pretrained("shyamsn97/Mario-GPT2-700-context-length")
    
    # 2. 建立 Dataset
    # 注意：這裡我們不需要傳入 prompter，因為我們只是要檢查 "文字" 修復結果
    # Dataset 內部的邏輯會處理 ID 修復，我們只要解碼 ID 即可
    dataset = MarioDataset(
        tokenizer=tokenizer,
        context_len=CONTEXT_LEN,
        height=HEIGHT,
    )

    print(f"📂 準備寫入檔案: {OUTPUT_FILE}")
    print(f"👀 將會連續讀取前 {CHECK_COUNT} 個視窗...")

    # 3. 開啟檔案準備寫入
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        
        # 🔥 關鍵：使用迴圈連續讀取 (0, 1, 2, 3...) 🔥
        for i in range(CHECK_COUNT):
            
            # 呼叫 __getitem__，這會觸發 process_window (修復水管邏輯)
            # 回傳格式是: (input_ids, attention_mask, hidden_states, hidden_mask)
            # 我們只需要第一個 input_ids
            # 改成只接收 2 個變數： (input_ids, attention_mask)
            # 我們只需要 input_ids，另一個 mask 用底線 _ 忽略
            input_ids, _ = dataset[i]
            
            # 解碼：把數字 ID 轉回字串 (這時是一條長長的字串)
            decoded_text = tokenizer.decode(input_ids)
            
            # --- 轉成 2D 網格 (視覺化) ---
            # 瑪利歐的資料是 "Column-Major" (一欄一欄接在一起的)
            # 我們要把它轉回 "Row-Major" (一行一行) 才能讓人眼看懂
            
            cols = []
            # 每 HEIGHT (14) 個字切成一欄
            for j in range(0, len(decoded_text), HEIGHT):
                col_str = decoded_text[j : j + HEIGHT]
                if len(col_str) == HEIGHT: # 確保長度正確
                    cols.append(col_str)
            
            # 轉置 (Transpose): 把每一欄的第 k 個字元組合成第 k 列
            rows = []
            for r in range(HEIGHT):
                row_str = ""
                for c in cols:
                    if r < len(c):
                        row_str += c[r]
                rows.append(row_str)
            
            full_map_str = "\n".join(rows)

            # --- 寫入檔案 ---
            f.write(f"=== Window {i} (Index {i}) ===\n")
            f.write(full_map_str)
            f.write("\n" + "="*50 + "\n\n") # 分隔線
            
            print(f"  -> 已處理 Window {i}")

    print(f"✅ 完成！請打開 {OUTPUT_FILE} 查看連續的地圖。")

if __name__ == "__main__":
    main()