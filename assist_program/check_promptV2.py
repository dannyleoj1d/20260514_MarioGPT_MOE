import torch
import os
import collections
import random
import numpy as np
from typing import List, Tuple, Dict
from mario_gpt import MarioLM, SampleOutput

# ==========================================
# 1. 全域設定區
# ==========================================
# 設定要執行生成的總次數
GENERATE_COUNT = 100

# MoE 模型參數 (必須與訓練時一致)
NUM_EXPERTS = 8
MOE_TOP_K = 2
TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
CHECKPOINT_DIR = "Mario-GPT2-700-context-length\iteration_99999"
BASE_MODEL = "random"

# 輸出資料夾名稱
OUTPUT_DIR = "multi_iter_test"

# Prompt 選項池
QUANTIFIERS = ["no", "little", "some", "many"]
ELEVATIONS = ["low", "high"]

# 統計計數器 (用於最後總結)
GLOBAL_STATS = {
    "pipe": {"correct": 0, "total": 0},
    "enemy": {"correct": 0, "total": 0},
    "block": {"correct": 0, "total": 0},
    "coin": {"correct": 0, "total": 0},  # New
    "rect": {"correct": 0, "total": 0},  # New
    "elevation": {"correct": 0, "total": 0}
}

# ==========================================
# 2. 定義 PromptJudge (地圖分析器)
# ==========================================
class PromptJudge:
    def __init__(self):
        # 更新後的統計閾值
        self.statistics = {
            "pipe": np.array([0.0, 1.0, 3.0]),
            "enemy": np.array([0.0, 1.0, 3.0]),
            "block": np.array([0.0, 7.0, 20.0]), # Updated
            "coin": np.array([0.0, 3.0, 8.0]),   # New
            "rect": np.array([0.0, 1.0, 2.0]),   # New
        }

    def get_keywords(self, key):
        # 統一使用這組關鍵字
        return ["no", "little", "some", "many"]

    # --- Property Getters (保留您程式碼風格) ---
    @property
    def rect_thresholds(self) -> Tuple[List[int], List[str]]:
        return self.statistics["rect"], self.get_keywords("rect")

    @property
    def coin_thresholds(self) -> Tuple[List[int], List[str]]:
        return self.statistics["coin"], self.get_keywords("coin")

    # --- Counting Functions ---
    def count_pipes(self, flattened_level: str) -> int:
        return flattened_level.count("<>")

    def count_enemies(self, flattened_level: str) -> int:
        return flattened_level.count("E") + flattened_level.count("B")

    def count_blocks(self, flattened_level: str) -> int:
        return np.sum([flattened_level.count(char) for char in ["S", "?", "Q"]])

    def count_coins(self, flattened_level: str) -> int:
        return flattened_level.count("o")

    def _find_rectangle_size(self, grid, row, col, H, W) -> Tuple[int, int]:
        """
        Helper: 檢查從 (row, col) 開始是否構成一個合法的 'T' 矩形。
        回傳 (height, width)，若無效回傳 (0, 0)。
        定義：邊界為 'T'，內部不含 'T' (通常為 '-')
        """
        # 1. 掃描上邊界寬度
        width = 0
        while col + width < W and grid[row][col + width] == 'T':
            width += 1
        if width < 2: return 0, 0 # 太窄

        # 2. 掃描左邊界高度
        height = 0
        while row + height < H and grid[row + height][col] == 'T':
            height += 1
        if height < 2: return 0, 0 # 太矮

        # 3. 檢查下邊界
        if row + height - 1 >= H: return 0, 0
        for c in range(col, col + width):
            if grid[row + height - 1][c] != 'T':
                return 0, 0

        # 4. 檢查右邊界
        if col + width - 1 >= W: return 0, 0
        for r in range(row, row + height):
            if grid[r][col + width - 1] != 'T':
                return 0, 0

        # 5. 檢查內部 (應為空或其他非T物件)
        # 簡單起見，我們只檢查是否完整閉合，不嚴格限制內部必須是 '-'
        # 但若題目要求內部必須是 '-'，可在這裡加檢查
        
        return height, width

    def count_rects(self, flattened_level: str) -> int:
        """Count complete rectangle structures in the level"""
        level_length = len(flattened_level)
        if level_length == 0: return 0
        
        # 假設高度固定為 14
        H = 14
        if level_length % H != 0:
            # 防呆：如果長度不對，嘗試容錯或回傳 0
            return 0
        
        W = level_length // H
        
        # 轉回 2D grid
        level_2d = []
        for i in range(H):
            start = i * W
            end = start + W
            level_2d.append(list(flattened_level[start:end]))
        
        rect_count = 0
        visited = set()
        
        for row in range(H):
            for col in range(W):
                # 找到潛在的左上角
                if (row, col) not in visited and level_2d[row][col] == 'T':
                    h, w = self._find_rectangle_size(level_2d, row, col, H, W)
                    
                    if h > 0 and w > 0:
                        rect_count += 1
                        # 標記整個矩形區域為 visited，避免重複計算
                        for r in range(row, row + h):
                            for c in range(col, col + w):
                                visited.add((r, c))
                                
        return rect_count

    # --- Labeling & Logic ---
    def get_label(self, category: str, count: int) -> str:
        if category not in self.statistics:
            return "unknown"
        thresholds = self.statistics[category]
        keywords = self.get_keywords(category)
        # digitize: return index where count would be inserted
        idx = np.digitize(count, thresholds, right=True)
        # 確保 idx 不超出 keywords 範圍 (例如 count 非常大時)
        idx = min(idx, len(keywords) - 1)
        return keywords[idx]

    def check_elevation(self, lines: List[str]) -> str:
        top_levels = lines[:6]
        for t in top_levels:
            if "X" in t or "<" in t or ">" in t:
                return "high"
        return "low"

    def parse_prompt(self, prompt_str: str) -> Dict[str, str]:
        parts = prompt_str.split(",")
        result = {}
        for part in parts:
            words = part.strip().split(" ")
            if len(words) >= 2:
                label = words[0]
                key = words[1]
                if "pipe" in key: result["pipe"] = label
                elif "enem" in key: result["enemy"] = label
                elif "block" in key: result["block"] = label
                elif "elev" in key: result["elevation"] = label
                elif "coin" in key: result["coin"] = label  # New
                elif "rect" in key: result["rect"] = label  # New
        return result

    def analyze_and_report(self, level_str: str, target_prompt: str) -> Tuple[str, Dict[str, bool]]:
        lines = [line for line in level_str.strip().split('\n') if line.strip()]
        flattened_level = "".join(lines)

        # 計算數值
        n_pipes = self.count_pipes(flattened_level)
        n_enemies = self.count_enemies(flattened_level)
        n_blocks = self.count_blocks(flattened_level)
        n_coins = self.count_coins(flattened_level) # New
        n_rects = self.count_rects(flattened_level) # New
        
        # 取得實際標籤
        actual_labels = {
            "pipe": self.get_label("pipe", n_pipes),
            "enemy": self.get_label("enemy", n_enemies),
            "block": self.get_label("block", n_blocks),
            "coin": self.get_label("coin", n_coins), # New
            "rect": self.get_label("rect", n_rects), # New
            "elevation": self.check_elevation(lines)
        }
        
        target_labels = self.parse_prompt(target_prompt)
        
        # 產生報告內容
        report_lines = []
        report_lines.append(f"使用的PROMPT: \"{target_prompt}\"")
        report_lines.append("")
        report_lines.append("="*50)
        report_lines.append("📊 關卡內容統計")
        report_lines.append("="*50)
        report_lines.append(f"🔧 水管 (Pipes):    {n_pipes:3d} -> [{actual_labels['pipe']}]")
        report_lines.append(f"👾 敵人 (Enemies):  {n_enemies:3d} -> [{actual_labels['enemy']}]")
        report_lines.append(f"🧱 磚塊 (Blocks):   {n_blocks:3d} -> [{actual_labels['block']}]")
        report_lines.append(f"💰 金幣 (Coins):    {n_coins:3d} -> [{actual_labels['coin']}]") # New
        report_lines.append(f"⬜ 矩形 (Rects):    {n_rects:3d} -> [{actual_labels['rect']}]") # New
        report_lines.append(f"⛰️  海拔 (Height):        -> [{actual_labels['elevation']}]")
        report_lines.append("-" * 50)
        
        generated_prompt_desc = (
            f"{actual_labels['pipe']} pipes, {actual_labels['enemy']} enemies, "
            f"{actual_labels['block']} blocks, {actual_labels['coin']} coins, "
            f"{actual_labels['rect']} rects, {actual_labels['elevation']} elevation"
        )
        report_lines.append(f"📝 實際對應 Prompt: \"{generated_prompt_desc}\"")
        report_lines.append("="*50)
        report_lines.append("")
        report_lines.append("🔍 差異比對結果:")
        report_lines.append("-" * 50)

        # 比對並紀錄結果
        check_order = ["pipe", "enemy", "block", "coin", "rect", "elevation"]
        results_bool = {}

        for key in check_order:
            if key not in target_labels: continue
            
            target_val = target_labels[key]
            actual_val = actual_labels[key]
            
            display_name = key.capitalize()
            
            if target_val == actual_val:
                status = "✅ [符合]"
                diff_msg = ""
                results_bool[key] = True
            else:
                status = "❌ [不符]"
                diff_msg = f"  (預期: {target_val} | 實際: {actual_val})"
                results_bool[key] = False
            
            report_lines.append(f"{status} {display_name:<10}: {actual_val:<10} {diff_msg}")

        report_lines.append("-" * 50)
        report_lines.append("\n\n") 
        
        return "\n".join(report_lines), results_bool

# ==========================================
# 3. 主程式邏輯
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # --- A. 初始化模型 ---
    print(f"Initializing MoE MarioGPT (Experts={NUM_EXPERTS}, TopK={MOE_TOP_K})...")
    mario_lm = MarioLM(
        lm_path=BASE_MODEL,       
        tokenizer_path=TOKENIZER_PATH,
        use_moe=True,             
        num_experts=NUM_EXPERTS,  
        moe_top_k=MOE_TOP_K       
    )

    safetensors_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
    bin_path = os.path.join(CHECKPOINT_DIR, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        print(f"Loading weights from {safetensors_path}...")
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path, device=str(mario_lm.device))
    elif os.path.exists(bin_path):
        print(f"Loading weights from {bin_path}...")
        state_dict = torch.load(bin_path, map_location=mario_lm.device)
    else:
        print(f"Error: Checkpoint not found in {CHECKPOINT_DIR}")
        exit()

    try:
        mario_lm.lm.load_state_dict(state_dict, strict=True)
        print("Success! Trained MoE weights loaded.")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        exit()

    if torch.cuda.is_available():
        mario_lm = mario_lm.to(torch.device('cuda'))
        print("Moved model to CUDA.")

    mario_lm.eval()
    judge = PromptJudge()
    report_filename = os.path.join(OUTPUT_DIR, "prompt_check_report.txt")
    
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("=== MOE MARIO GPT BATCH GENERATION REPORT ===\n\n")

    # --- B. 迴圈生成 ---
    print(f"\nStarting batch generation for {GENERATE_COUNT} iterations...\n")

    for i in range(GENERATE_COUNT):
        print(f"--- Iteration {i+1}/{GENERATE_COUNT} ---")
        
        # 1. 隨機產生 Prompt (加入 coin 和 rect)
        p_pipe = random.choice(QUANTIFIERS)
        p_enemy = random.choice(QUANTIFIERS)
        p_block = random.choice(QUANTIFIERS)
        p_coin = random.choice(QUANTIFIERS) # New
        p_rect = random.choice(QUANTIFIERS) # New
        p_elev = random.choice(ELEVATIONS)
        
        # 更新 Prompt 格式以包含新特徵
        current_prompt = (
            f"{p_pipe} pipes, {p_enemy} enemies, {p_block} blocks, "
            f"{p_coin} coins, {p_rect} rects, {p_elev} elevation"
        )
        prompts = [current_prompt]
        print(f"Prompt: {current_prompt}")

        # 2. 生成關卡
        generated_level = mario_lm.sample(
            prompts=prompts,
            num_steps=1400,
            temperature=2.0, 
            use_tqdm=True
        )

        # 3. 儲存
        img_filename = os.path.join(OUTPUT_DIR, f"moe_generated_level_{i}.png")
        txt_filename = os.path.join(OUTPUT_DIR, f"moe_generated_level_{i}.txt")
        
        generated_level.img.save(img_filename)
        generated_level.save(txt_filename)
        
        with open(txt_filename, 'r') as f:
            level_content = f.read()

        with open(txt_filename, "a", encoding="utf-8") as f:
            f.write("\n\n" + f'使用的PROMPT: "{current_prompt}"')
            
        print(f"Saved: {img_filename}, {txt_filename}")

        # 4. 判斷與報告
        report_text, results_bool = judge.analyze_and_report(level_content, current_prompt)
        
        with open(report_filename, "a", encoding="utf-8") as f:
            f.write(f"--- [Run #{i+1}] ---\n")
            f.write(report_text)
        
        # 5. 更新統計
        for key in results_bool:
            GLOBAL_STATS[key]["total"] += 1
            if results_bool[key]:
                GLOBAL_STATS[key]["correct"] += 1
        
        print("Analysis completed.\n")

    # --- C. 最終統計輸出 ---
    print("="*40)
    print("       FINAL ACCURACY STATISTICS       ")
    print("="*40)
    
    summary_lines = []
    summary_lines.append("\n=== FINAL STATISTICS SUMMARY ===\n")
    
    # 加入 coin 和 rect 到總結
    for key in ["pipe", "enemy", "block", "coin", "rect", "elevation"]:
        stats = GLOBAL_STATS[key]
        total = stats["total"]
        correct = stats["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        line = f"{key.capitalize():<10}: {correct}/{total} correct ({accuracy:.1f}%)"
        print(line)
        summary_lines.append(line)

    with open(report_filename, "a", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\nFull report saved to {report_filename}")

if __name__ == "__main__":
    main()