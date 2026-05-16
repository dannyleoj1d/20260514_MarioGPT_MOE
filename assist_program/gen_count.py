import numpy as np

# ==========================================
# 1. 設定閾值 (Thresholds)
# 這裡使用你目前的設定 (嚴格版)，方便你對照現狀
# ==========================================
STATISTICS = {
    # 閾值: [Q33, Q66, Q95]
    # 邏輯: 
    #   x <= Q33      -> Index 0 (No/Little)
    #   Q33 < x <= Q66 -> Index 1 (Little/Some)
    #   ...
    "enemy": np.array([1.0, 3.0, 7.0]),
    "pipe": np.array([0.0, 2.0, 5.0]),
    "block": np.array([50.0, 75.0, 176.0]),
    "coin": np.array([0.0, 2.0, 12.0]),
}

# 關鍵字對應表 (修正了原本 Block 重複 "little" 的問題)
KEYWORDS = {
    "enemy": ["no", "little", "some", "many"],
    "pipe":  ["no", "little", "some", "many"],
    "block": ["no", "little", "some", "many"], # 修正後
    "coin":  ["no", "little", "some", "many"],
}

# ==========================================
# 2. 統計函式 (Counting Logic)
# ==========================================
def count_pipes(flattened_level: str) -> int:
    return flattened_level.count("<>")

def count_enemies(flattened_level: str) -> int:
    return flattened_level.count("E") + flattened_level.count("B")

def count_blocks(flattened_level: str) -> int:
    return np.sum([flattened_level.count(char) for char in ["X", "S", "?", "Q"]])

def count_coins(flattened_level: str) -> int:
    return flattened_level.count("o")

def check_elevation(level_lines) -> str:
    # 檢查前 6 行 (高空區域)
    top_levels = level_lines[:6]
    flattened_top = "".join(top_levels)
    
    # 只要有 X(磚塊) 或 水管(<, >) 就判定為 High
    if "X" in flattened_top or "<" in flattened_top or ">" in flattened_top:
        return "high"
    return "low"

# ==========================================
# 3. 分類核心 (Classifier)
# ==========================================
def classify(feature_name, count):
    thresholds = STATISTICS[feature_name]
    keywords = KEYWORDS[feature_name]
    
    # np.digitize(right=True): x <= threshold
    # 回傳值:
    # 0: count <= thresholds[0]
    # 1: thresholds[0] < count <= thresholds[1]
    # 2: thresholds[1] < count <= thresholds[2]
    # 3: count > thresholds[2]
    idx = np.digitize(count, thresholds, right=True)
    
    label = keywords[idx] if idx < len(keywords) else keywords[-1]
    
    return label, idx, thresholds

# ==========================================
# 4. 主程式
# ==========================================
def analyze_level(level_str):
    # 預處理：移除空白行，轉成列表
    level_lines = [line for line in level_str.strip().split("\n") if line.strip()]
    flattened_level = "".join(level_lines)
    
    print("-" * 60)
    print(f"地圖尺寸: {len(level_lines)} 行 x {len(level_lines[0])} 列")
    print("-" * 60)

    # 1. 執行統計
    counts = {
        "pipe": count_pipes(flattened_level),
        "enemy": count_enemies(flattened_level),
        "block": count_blocks(flattened_level),
        "coin": count_coins(flattened_level),
    }
    
    elevation = check_elevation(level_lines)

    # 2. 執行分類並輸出報告
    print(f"{'Feature':<10} | {'Count':<6} | {'Label':<8} | {'Interval Logic (Debug)':<30}")
    print("-" * 60)
    
    for feat, count in counts.items():
        label, idx, th = classify(feat, count)
        
        # 產生區間說明字串
        if idx == 0:
            interval = f"x <= {th[0]}"
        elif idx == 1:
            interval = f"{th[0]} < x <= {th[1]}"
        elif idx == 2:
            interval = f"{th[1]} < x <= {th[2]}"
        else:
            interval = f"x > {th[2]}"
            
        print(f"{feat:<10} | {count:<6} | {label:<8} | {interval:<30}")

    print(f"{'elevation':<10} | {'-':<6} | {elevation:<8} | Check top 6 rows for X/< />")
    print("-" * 60)

# ==========================================
# 測試資料 (你的地圖)
# ==========================================
user_level = """
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------SS----SSSSXSSSSSSS------------------
--------------------------------------------------------------------------X-------------X-----------
--------------------------------------------------------------------------X------X------------------
-----o--------------o-o-------------------------------------------S-S----SX------X------------------
----------------------------------------------------------------------<>--X-X------X----------------
----------------------------------------------------------<>----------[]--X-XX-X--X-----------------
--------------<>---------->-------------------------------[]-E--------[]--X-X-----------------<>X---
--------------[]----------]-------------------------------[]E---------[]--X-X---------------X-[]X---
-XXXXXXXXXXXXXXXXXXXXXXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-X---------------XXXXXXXXXX
"""

if __name__ == "__main__":
    # 移除第一行的 Prompt 文字，只留地圖部分
    map_only = "\n".join(user_level.strip().split("\n")[1:])
    analyze_level(map_only)