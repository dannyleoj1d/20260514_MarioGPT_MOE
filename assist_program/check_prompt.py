import numpy as np
from typing import List, Tuple, Dict

# ==========================================
# 1. 設定：輸入你的關卡與原始 Prompt
# ==========================================
RAW_LEVEL = """
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------XXXXXXXXX-------
-------------------------------------------------------------------------------------------[]-------
-------------------------------------------------------------------------------------------[]-------
--------------------------------------------------X-X----------------------X---------------[]-------
Q------------------------------------------XX--X-------SS-------------------S--------------[]-------
-X------------------------------E---------XXX--X-------------------------------------------<--------
-X---<>------B--------<>-----------SS----XXXX--X-----------------------------S--------------E-------
EX---[]---------------[]-----------SS---XXXXX--X--------------------E--------S---------------E--E---
XX---XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXSXXXXXXXXX--X------XXXXXXXXXXEXX<>XXXXXXXXS----XXXXXXXX--EXXXXXXX
"""

# 設定原本預期的 Prompt (Target)
ORIGIN_PROMPT = "some pipes, no enemies, some blocks, high elevation"

# ==========================================
# 2. 定義統計數據
# ==========================================
STATISTICS = {
    "enemy": np.array([1.0, 3.0, 7.0]),
    "pipe": np.array([0.0, 2.0, 5.0]),
    "block": np.array([50.0, 75.0, 176.0]),
}

class PromptJudge:
    def __init__(self):
        self.statistics = STATISTICS

    def get_keywords(self, key):
        if key == "block":
            return ["little", "little", "some", "many"]
        return ["no", "little", "some", "many"]

    def count_pipes(self, flattened_level: str) -> int:
        return flattened_level.count("<>")

    def count_enemies(self, flattened_level: str) -> int:
        # 計算 E (敵人) 和 B (砲彈)
        return flattened_level.count("E") + flattened_level.count("B")

    def count_blocks(self, flattened_level: str) -> int:
        # 計算各種磚塊
        return np.sum([flattened_level.count(char) for char in ["X", "S", "?", "Q"]])

    def get_label(self, category: str, count: int) -> str:
        thresholds = self.statistics[category]
        keywords = self.get_keywords(category)
        idx = np.digitize(count, thresholds, right=True)
        return keywords[idx]

    def check_elevation(self, lines: List[str]) -> str:
        # 檢查前 6 行 (高空區域)
        top_levels = lines[:6]
        for t in top_levels:
            if "X" in t or "<" in t or ">" in t:
                return "high"
        return "low"

    def parse_prompt(self, prompt_str: str) -> Dict[str, str]:
        """將 Prompt 字串解析為字典，方便比對"""
        # 預期格式: "many pipes, some enemies, some blocks, low elevation"
        parts = prompt_str.split(",")
        result = {}
        for part in parts:
            words = part.strip().split(" ")
            if len(words) >= 2:
                label = words[0]  # e.g., "many", "no", "low"
                key = words[1]    # e.g., "pipes", "enemies", "elevation"
                
                # 標準化 key 名稱
                if "pipe" in key: result["pipe"] = label
                elif "enem" in key: result["enemy"] = label
                elif "block" in key: result["block"] = label
                elif "elev" in key: result["elevation"] = label
        return result

    def compare_prompts(self, origin: Dict[str, str], actual: Dict[str, str]):
        """比較並印出詳細差異"""
        print("\n🔍 差異比對結果 (Target vs Actual):")
        print("-" * 50)
        
        # 定義檢查順序
        check_order = ["pipe", "enemy", "block", "elevation"]
        is_perfect = True

        for key in check_order:
            if key not in origin or key not in actual:
                continue

            target_val = origin[key]
            actual_val = actual[key]
            
            # 美化顯示名稱
            display_name = key.capitalize()
            if key == "pipe": display_name = "Pipes"
            elif key == "enemy": display_name = "Enemies"
            elif key == "block": display_name = "Blocks"

            if target_val == actual_val:
                status = "✅ [符合]"
                diff_msg = ""
            else:
                status = "❌ [不符]"
                diff_msg = f"  (預期: {target_val} | 實際: {actual_val})"
                is_perfect = False

            print(f"{status} {display_name:<10}: {actual_val:<10} {diff_msg}")

        print("-" * 50)
        if is_perfect:
            print("🏆 結果: 完美符合 Prompt 要求！")
        else:
            print("⚠️ 結果: 部分特徵未達標，請參考上述差異。")

    def analyze(self, raw_level_str: str, origin_prompt_str: str):
        # 1. 前處理
        lines = [line for line in raw_level_str.strip().split('\n') if line.strip()]
        print(f"DEBUG: 偵測到地圖高度: {len(lines)}")
        flattened_level = "".join(lines)

        # 2. 計算實際數值
        n_pipes = self.count_pipes(flattened_level)
        n_enemies = self.count_enemies(flattened_level)
        n_blocks = self.count_blocks(flattened_level)
        
        # 3. 取得實際標籤
        actual_labels = {
            "pipe": self.get_label("pipe", n_pipes),
            "enemy": self.get_label("enemy", n_enemies),
            "block": self.get_label("block", n_blocks),
            "elevation": self.check_elevation(lines)
        }

        # 4. 輸出基礎統計報告
        print("="*50)
        print("📊 關卡內容統計")
        print("="*50)
        print(f"🔧 水管 (Pipes):   {n_pipes:3d} 個 -> [{actual_labels['pipe']} pipes]")
        print(f"👾 敵人 (Enemies): {n_enemies:3d} 隻 -> [{actual_labels['enemy']} enemies]")
        print(f"🧱 磚塊 (Blocks):  {n_blocks:3d} 個 -> [{actual_labels['block']} blocks]")
        print(f"⛰️  海拔 (Height):        -> [{actual_labels['elevation']} elevation]")
        
        generated_prompt = f"{actual_labels['pipe']} pipes, {actual_labels['enemy']} enemies, {actual_labels['block']} blocks, {actual_labels['elevation']} elevation"
        print("-" * 50)
        print(f"📝 實際生成的 Prompt: \"{generated_prompt}\"")
        print("="*50)

        # 5. 進行比對
        target_labels = self.parse_prompt(origin_prompt_str)
        self.compare_prompts(target_labels, actual_labels)

if __name__ == "__main__":
    judge = PromptJudge()
    judge.analyze(RAW_LEVEL, ORIGIN_PROMPT)