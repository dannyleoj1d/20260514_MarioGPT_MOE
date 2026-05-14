import os
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer

# 引入你的 Dataset
from mario_gpt.dataset import MarioDataset

# ================= 設定區 =================
OUTPUT_FILE = "prompt_statistics_report.txt"
CONTEXT_LEN = 700   # 必須跟 dataset 設定一致
HEIGHT = 14         # 瑪利歐高度
# =========================================

# 來自 prompter.py 的統計門檻值
STATISTICS = {
    "enemy": np.array([1.0, 3.0, 7.0]),
    "pipe": np.array([0.0, 2.0, 5.0]),
    "block": np.array([50.0, 75.0, 176.0]),
    "coin": np.array([0.0, 3.0, 8.0]),
    "rect": np.array([0.0, 1.0, 2.0]),
}

class LightweightAnalyzer:
    """
    輕量級分析器：不載入 BART 模型，只負責計算數量與對應的標籤。
    """
    def __init__(self):
        self.statistics = STATISTICS

    def get_keywords(self, key):
        if key == "block":
            return ["little", "little", "some", "many"]
        return ["no", "little", "some", "many"]

    def count_pipes(self, flattened_level: str) -> int:
        return flattened_level.count("<>")

    def count_enemies(self, flattened_level: str) -> int:
        return flattened_level.count("E") + flattened_level.count("B")

    def count_blocks(self, flattened_level: str) -> int:
        return np.sum([flattened_level.count(char) for char in ["X", "S", "?", "Q"]])

    def count_coins(self, flattened_level: str) -> int:
        return flattened_level.count("o")

    def count_rects(self, cols: list, height: int) -> int:
        W = len(cols)
        H = height
        grid = [[cols[c][r] if r < len(cols[c]) else '-' for c in range(W)] for r in range(H)]
        rect_count = 0
        visited = set()
        for row in range(H):
            for col in range(W):
                if (row, col) not in visited and grid[row][col] == 'T':
                    width = 0
                    while col + width < W and grid[row][col + width] == 'T':
                        width += 1
                    h = 0
                    while row + h < H and grid[row + h][col] == 'T':
                        h += 1
                    if width < 2 or h < 2:
                        continue
                    valid = all(grid[row + h - 1][c] == 'T' for c in range(col, col + width))
                    valid = valid and all(grid[r][col + width - 1] == 'T' for r in range(row, row + h))
                    if valid:
                        rect_count += 1
                        for r in range(row, row + h):
                            for c in range(col, col + width):
                                visited.add((r, c))
        return rect_count

    def get_label(self, category: str, count: int) -> str:
        thresholds = self.statistics[category]
        keywords = self.get_keywords(category)
        # 使用 numpy 的 digitize 找區間 (跟 prompter.py 邏輯一致)
        idx = np.digitize(count, thresholds, right=True)
        return keywords[idx]

    def check_elevation(self, rows: list[str]) -> str:
        # 檢查前 6 行 (高空區域)
        top_levels = rows[:6]
        for t in top_levels:
            if "X" in t or "<" in t or ">" in t:
                return "high"
        return "low"

    def analyze_slice(self, level_str: str, height: int):
        """
        輸入: tokenizer 解碼後的一維長字串 (Column-Major)
        輸出: 完整的 prompt 字串
        """
        # 1. 重建成 2D 網格 (Rows) 以便計算 Elevation
        # level_str 是 column-major, bottom-to-top：每欄 index 0 = 地板, index 13 = 天空
        cols = [level_str[i:i+height] for i in range(0, len(level_str), height)]

        # 先建出 bottom-first rows，再反轉為 top-first（row 0 = 天空, row 13 = 地板）
        rows = []
        for r in range(height):
            row_str = ""
            for c in cols:
                if r < len(c):
                    row_str += c[r]
            rows.append(row_str)
        rows = rows[::-1]  # 反轉：讓 rows[0]=sky, rows[13]=ground，與 prompter 一致

        # 2. 準備壓扁字串 (計算物件數量用)
        flattened_level = "".join(rows)

        # 3. 計算各項數值
        n_pipes = self.count_pipes(flattened_level)
        n_enemies = self.count_enemies(flattened_level)
        n_blocks = self.count_blocks(flattened_level)
        n_coins = self.count_coins(flattened_level)
        n_rects = self.count_rects(cols, height)

        # 4. 取得標籤
        pipe_tag = self.get_label("pipe", n_pipes)
        enemy_tag = self.get_label("enemy", n_enemies)
        block_tag = self.get_label("block", n_blocks)
        coin_tag = self.get_label("coin", n_coins)
        rect_tag = self.get_label("rect", n_rects)
        elev_tag = self.check_elevation(rows)

        # 5. 組合 Prompt
        prompt = f"{pipe_tag} pipes, {enemy_tag} enemies, {block_tag} blocks, {coin_tag} coins, {rect_tag} rects, {elev_tag} elevation"
        return prompt

def main():
    print("🚀 初始化 Tokenizer 與 Dataset (這可能需要一點時間載入)...")
    
    tokenizer = AutoTokenizer.from_pretrained("shyamsn97/Mario-GPT2-700-context-length")
    
    # 這裡我們利用 MarioDataset 的切片邏輯
    dataset = MarioDataset(
        tokenizer=tokenizer,
        context_len=CONTEXT_LEN,
        height=HEIGHT,
        # 不需要 prompter 參數，因為我們自己算
    )
    
    total_slices = len(dataset)
    print(f"📊 總共有 {total_slices} 個切片需要統計。")
    print("⏳ 開始掃描與統計 (包含水管修復邏輯)...")

    analyzer = LightweightAnalyzer()
    stats_counter = Counter()
    feature_counters = {
        "pipe":   Counter(),
        "enemy":  Counter(),
        "block":  Counter(),
        "coin":   Counter(),
        "rect":   Counter(),
        "elevation": Counter(),
    }

    # 使用 tqdm 顯示進度條
    for i in tqdm(range(total_slices), desc="Analyzing"):
        input_ids, _, _ = dataset[i]
        level_str = tokenizer.decode(input_ids.tolist())
        prompt = analyzer.analyze_slice(level_str, HEIGHT)
        stats_counter[prompt] += 1

        # 拆解各特徵標籤
        parts = [p.strip() for p in prompt.split(",")]
        for part in parts:
            words = part.split()
            if len(words) >= 2:
                label, key = words[0], words[1]
                if "pipe"  in key: feature_counters["pipe"][label]      += 1
                elif "enem" in key: feature_counters["enemy"][label]    += 1
                elif "block" in key: feature_counters["block"][label]   += 1
                elif "coin" in key: feature_counters["coin"][label]     += 1
                elif "rect" in key: feature_counters["rect"][label]     += 1
                elif "elev" in key: feature_counters["elevation"][label]+= 1

    # ================= 產生報告 =================
    print(f"\n💾 正在寫入報告至 {OUTPUT_FILE} ...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("           MARIO DATASET PROMPT STATISTICS        \n")
        f.write("==================================================\n")
        f.write(f"Total Slices Scanned: {total_slices}\n\n")

        # --- 各特徵個別分佈 ---
        f.write("==================================================\n")
        f.write("          PER-FEATURE LABEL DISTRIBUTION          \n")
        f.write("==================================================\n")
        for feature, counter in feature_counters.items():
            f.write(f"\n[{feature.upper()}]\n")
            for label in ["no", "little", "some", "many", "low", "high"]:
                if label in counter:
                    pct = counter[label] / total_slices * 100
                    bar = "█" * int(pct / 2)
                    f.write(f"  {label:<8}: {counter[label]:5d} ({pct:5.1f}%) {bar}\n")

        # --- 完整 prompt 組合分佈 ---
        f.write("\n==================================================\n")
        f.write("           FULL PROMPT COMBINATION RANKING        \n")
        f.write("==================================================\n")
        for prompt, count in stats_counter.most_common():
            percentage = (count / total_slices) * 100
            f.write(f'"{prompt}" 佔 {percentage:6.2f}% (Count: {count})\n')

    print("✅ 完成！")
    print(f"👉 請查看報告檔案: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()