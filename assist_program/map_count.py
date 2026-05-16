import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy import stats  # 引入統計模組

# 確保可以導入模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mario_gpt.dataset import MarioDataset
from mario_gpt import Prompter
from transformers import AutoTokenizer

# 設定模型路徑
DEFAULT_MODEL = "distilgpt2"

def reconstruct_level(flattened_str, height=14):
    """
    將 MarioGPT 的直向訓練資料 (Column-Major) 轉回 橫向視覺地圖 (Row-Major)。
    這樣 count_pipes 才能正確偵測到 "<>"。
    """
    if len(flattened_str) == 0:
        return ""
    
    # 訓練資料是把一列一列(Column)接起來的，所以我們每 14 個字切成一列
    cols = [flattened_str[i:i+height] for i in range(0, len(flattened_str), height)]
    
    if not cols: return ""
    
    rows = []
    real_height = len(cols[0])
    for r in range(real_height):
        row_str = ""
        for c in range(len(cols)):
            if r < len(cols[c]):
                row_str += cols[c][r]
        rows.append(row_str)
        
    return "".join(rows)

def main():
    print("🚀 初始化 Base Tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    print("📂 載入 MarioDataset (並自動訓練專屬 Tokenizer)...")
    dataset = MarioDataset(tokenizer=base_tokenizer, context_len=700)
    
    print("🏷️ 初始化 Prompter (使用 Dataset 訓練好的 Tokenizer)...")
    prompter = Prompter(dataset.tokenizer)

    print(f"\n📊 開始分析 {len(dataset)} 筆資料的分佈...")
    
    # 用來存標籤統計 (原本的功能)
    label_stats = {
        "pipe": Counter(),
        "enemy": Counter(),
        "block": Counter(),
        "coin": Counter(),
        "rect": Counter(),
        "elevation": Counter()
    }

    # 用來存原始數量 (新功能: 用於計算 Quantiles)
    raw_counts = defaultdict(list)

    # 遍歷整個 Dataset
    for i in tqdm(range(len(dataset)), desc="Scanning"):
        # 1. 取得資料
        item = dataset[i]
        
        if isinstance(item, dict):
            input_ids = item["input_ids"]
        elif isinstance(item, (list, tuple)):
            input_ids = item[0]
        else:
            continue

        # 2. 解碼 (Decode)
        flattened_col_major = dataset.tokenizer.decode(input_ids)
        
        # 3. 重組地圖 (Transpose Back)
        visual_level = reconstruct_level(flattened_col_major, height=14)

        # 4. 呼叫 Prompter 取得原始數量 (Raw Counts)
        # 注意: 請確保你的 Prompter 類別中有定義 count_rects 和 count_rects2
        # 如果是原版 MarioGPT，可能沒有 count_rects，若報錯請註解掉相關行數
        try:
            n_pipe = prompter.count_pipes(visual_level)
            n_enemy = prompter.count_enemies(visual_level)
            n_block = prompter.count_blocks(visual_level)
            n_coin = prompter.count_coins(visual_level)
            
            # 嘗試計算 rects (依賴於自定義的 Prompter)
            n_rect = prompter.count_rects(visual_level) if hasattr(prompter, 'count_rects') else 0

            # 存入列表
            raw_counts["pipe"].append(n_pipe)
            raw_counts["enemy"].append(n_enemy)
            raw_counts["block"].append(n_block)
            raw_counts["coin"].append(n_coin)
            raw_counts["rect"].append(n_rect)

        except Exception as e:
            # 容錯處理，避免因為缺少某個 count method 而中斷
            pass

        # 5. 取得標籤 (原本的功能)
        _, pipe_lbl = prompter.pipe_prompt(visual_level, visual_level)
        _, enemy_lbl = prompter.enemy_prompt(visual_level, visual_level)
        _, block_lbl = prompter.block_prompt(visual_level, visual_level)
        _, coin_lbl = prompter.coin_prompt(visual_level, visual_level)
        _, elev_lbl = prompter.elevation_prompt(visual_level, visual_level)
        _, rect_lbl = prompter.rect_prompt(visual_level, visual_level)

        label_stats["pipe"][pipe_lbl] += 1
        label_stats["enemy"][enemy_lbl] += 1
        label_stats["block"][block_lbl] += 1
        label_stats["coin"][coin_lbl] += 1
        label_stats["rect"][rect_lbl] += 1
        label_stats["elevation"][elev_lbl] += 1

    # ================= 輸出原始標籤分佈 =================
    print("\n" + "="*50)
    print(f"📈 Dataset 標籤分佈統計 (修正版) (總數: {len(dataset)})")
    print("="*50)

    order = ["no", "little", "some", "many"]
    elev_order = ["low", "high"]

    for feature, counter in label_stats.items():
        print(f"\n🔹 Feature: {feature.upper()}")
        print(f"{'Label':<10} | {'Count':<8} | {'Percentage':<10}")
        print("-" * 35)
        
        current_order = elev_order if feature == "elevation" else order
        total = sum(counter.values())
        
        for label in current_order:
            count = counter.get(label, 0)
            pct = (count / total) * 100 if total > 0 else 0.0
            marker = "⚠️" if pct < 1.0 else "" 
            print(f"{label:<10} | {count:<8} | {pct:>6.2f}% {marker}")

    # ================= 計算並輸出 Quantiles (新功能) =================
    print("\n" + "="*60)
    print("🎯 計算出的分位數閥值 (Calculated Thresholds)")
    print("這些數值代表了該資料集 [33%, 66%, 95%] 的分界點")
    print("可以直接複製下方的 dictionary 到你的程式碼中使用")
    print("="*60)

    d = {}
    
    # 確保列表不為空，避免報錯
    if raw_counts["pipe"]:
        d["pipe"] = stats.mstats.mquantiles(raw_counts["pipe"], [0.33, 0.66, 0.95])
    if raw_counts["enemy"]:
        d["enemy"] = stats.mstats.mquantiles(raw_counts["enemy"], [0.33, 0.66, 0.95])
    if raw_counts["block"]:
        d["block"] = stats.mstats.mquantiles(raw_counts["block"], [0.33, 0.66, 0.95])
    if raw_counts["coin"]:
        d["coin"] = stats.mstats.mquantiles(raw_counts["coin"], [0.33, 0.66, 0.95])
    
    # Rect 可能在原版沒有，這裡做檢查
    if raw_counts["rect"] and sum(raw_counts["rect"]) > 0:
        d["rect"] = stats.mstats.mquantiles(raw_counts["rect"], [0.33, 0.66, 0.95])
    else:
        d["rect"] = np.array([0., 0., 0.]) # 預設值
        
    if raw_counts["rect2"] and sum(raw_counts["rect2"]) > 0:
        d["rect2"] = stats.mstats.mquantiles(raw_counts["rect2"], [0.33, 0.66, 0.95])
    else:
        d["rect2"] = np.array([0., 0., 0.]) # 預設值

    # 漂亮列印
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    print("STATISTICS = {")
    for key, val in d.items():
        # 轉成列表顯示比較好看，並保留小數點
        formatted_val = [round(x, 2) for x in val]
        print(f'    "{key}": np.array({formatted_val}),')
    print("}")

if __name__ == "__main__":
    main()