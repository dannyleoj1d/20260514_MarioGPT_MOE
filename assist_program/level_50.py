import sys

# 嘗試匯入 mario_gpt，如果沒安裝則報錯
try:
    from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS
except ImportError:
    print("[Error] 請先安裝 mario-gpt: pip install mario-gpt")
    sys.exit(1)

def extract_first_n_columns(level_str: str, n: int = 50) -> str:
    """
    擷取地圖字串的前 n 個直行。
    """
    # 1. 移除前後空白並按換行符號拆分成列表
    lines = level_str.strip().split('\n')
    
    # 2. 對每一行進行切片，只取前 n 個字元
    #    line[:n] 會從索引 0 取到 n-1
    sliced_lines = [line[:n] for line in lines]
    
    # 3. 將處理後的行重新組合成字串
    return "\n".join(sliced_lines)

# === 主執行區 ===
if __name__ == "__main__":
    # 執行擷取
    column_limit = 100
    short_level = extract_first_n_columns(FULL_LEVEL_STR_WITH_PATHS, column_limit)
    
    # 1. 直接印在螢幕上預覽
    print(f"--- First {column_limit} Columns ---")
    print(short_level)
    print("--------------------------------")

    # 2. (選用) 存成檔案以便查看
    output_filename = "first_50_cols.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(short_level)
    
    print(f"已儲存至 {output_filename}")