import os
import numpy as np
import random
from typing import List, Tuple, Optional

# ==========================================
# 1. 載入原始地圖
# ==========================================
try:
    from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS
except ImportError:
    print("[WARNING] Cannot import 'FULL_LEVEL_STR_WITH_PATHS'. Using dummy map.")
    # 建立一個足夠大的空白地圖供測試
    FULL_LEVEL_STR_WITH_PATHS = ("-" * 1400)

# ==========================================
# 2. 矩形生成器 (X-Rect Generator)
# ==========================================
class RectInjector:
    def __init__(self, level_str: str):
        self.lines = [line.strip() for line in level_str.strip().split('\n') if line.strip()]
        self.height = 14
        # 確保地圖是完整的矩形
        max_width = max(len(line) for line in self.lines)
        self.width = max_width
        # 轉成 numpy array 方便操作
        self.grid = np.array([list(line.ljust(max_width, '-')) for line in self.lines])
        self.placed_count = 0

    def is_area_clear(self, r: int, c: int, h: int, w: int, padding: int = 2) -> bool:
        """
        檢查區域是否合法：
        1. 包含 padding 的範圍內必須全是 '-' (空氣)
        2. 不能超出地圖邊界
        """
        # 計算檢查範圍 (包含 padding)
        # 注意邊界檢查，不能小於 0 或大於地圖尺寸
        r_start = max(0, r - padding)
        r_end = min(self.height, r + h + padding)
        c_start = max(0, c - padding)
        c_end = min(self.width, c + w + padding)

        # 取出子區域
        subgrid = self.grid[r_start:r_end, c_start:c_end]

        # 核心條件：區域內必須全部都是 '-'
        # 這樣保證了不會覆蓋其他結構，且周圍有足夠的 padding
        return np.all(subgrid == '-')

    def draw_x_rect(self, r: int, c: int, h: int, w: int):
        """
        繪製帶有 X 結構的矩形
        """
        for i in range(h):
            for j in range(w):
                # 1. 繪製邊框
                is_border = (i == 0 or i == h - 1 or j == 0 or j == w - 1)
                
                # 2. 繪製 X 結構 (對角線)
                # 使用簡單的線性比例判斷
                # 左上到右下: i / (h-1) approx j / (w-1)
                # 左下到右上: i / (h-1) approx (w-1-j) / (w-1)
                
                # 容許誤差值，讓線條在小矩形中比較明顯
                threshold = 0.8
                
                # 正規化座標 (0.0 ~ 1.0)
                y_norm = i / (h - 1)
                x_norm = j / (w - 1)
                
                # 主對角線
                is_diag1 = abs(y_norm - x_norm) * (w-1) < threshold
                # 副對角線
                is_diag2 = abs(y_norm - (1 - x_norm)) * (w-1) < threshold

                if is_border or is_diag1 or is_diag2:
                    self.grid[r + i, c + j] = 'T'

    def max_fill(self):
        """
        嘗試塞入所有可塞入的地方
        """
        print("⚡ Starting Max Fill Process...")
        attempts = 0
        max_attempts = 2000 # 嘗試次數上限，越高填得越滿
        
        while attempts < max_attempts:
            attempts += 1
            
            # 隨機大小 8 到 4
            h = random.randint(4, 8)
            w = random.randint(4, 8)
            
            # 邊界檢查
            if self.height - h <= 0 or self.width - w <= 0: continue

            # 隨機位置
            r = random.randint(0, self.height - h)
            c = random.randint(0, self.width - w)

            # 檢查條件：周圍須有兩行的 - TOKEN 包覆
            if self.is_area_clear(r, c, h, w, padding=2):
                self.draw_x_rect(r, c, h, w)
                self.placed_count += 1
                # print(f"   [+] Placed rect at ({r},{c}) size {h}x{w}")
        
        print(f"✅ Max fill complete. Total placed: {self.placed_count}")

    def save_to_file(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            for row in self.grid:
                f.write("".join(row) + "\n")
        print(f"💾 File saved to: {filename}")

    def get_flattened_content(self) -> str:
        return "".join(["".join(row) for row in self.grid])

# ==========================================
# 3. 驗證器 (整合您提供的 count_rects)
# ==========================================
class RectValidator:
    def _find_rectangle_size(self, grid, row, col, H, W) -> int:
        """
        輔助函式：檢查從 (row, col) 開始的邊框是否構成完整矩形。
        回傳矩形較大邊長以便計數，若無效回傳 0。
        """
        # 1. 掃描上邊界寬度
        width = 0
        while col + width < W and grid[row][col + width] == 'T':
            width += 1
        if width < 4: return 0 # 題目設定最小為 4

        # 2. 掃描左邊界高度
        height = 0
        while row + height < H and grid[row + height][col] == 'T':
            height += 1
        if height < 4: return 0

        # 3. 檢查下邊界
        if row + height - 1 >= H: return 0
        for c in range(col, col + width):
            if grid[row + height - 1][c] != 'T':
                return 0

        # 4. 檢查右邊界
        if col + width - 1 >= W: return 0
        for r in range(row, row + height):
            if grid[r][col + width - 1] != 'T':
                return 0

        # 5. 內部檢查
        # 因為您的需求是 "矩形中間的空間會有T TOKEN構成的X結構"
        # 所以這裡只要邊框完整，我們就視為有效矩形，忽略內部是否有 T
        
        return max(width, height)

    def count_rects(self, flattened_level: str) -> int:
        """Count complete rectangle structures in the level"""
        level_length = len(flattened_level)
        if level_length % 14 != 0:
            print('LLLLLLLLLLL', f'level_length={level_length}')
            return 0 
        
        width = level_length // 14
        level_2d = []
        for i in range(14):
            start = i * width
            end = start + width
            level_2d.append(list(flattened_level[start:end]))
        
        rect_count = 0
        visited = set()
        
        # Scan for rectangle structures
        for row in range(14):
            for col in range(width):
                if (row, col) not in visited and level_2d[row][col] == 'T':
                    
                    # Check if this 'T' is the top-left corner of a rectangle
                    rect_size = self._find_rectangle_size(level_2d, row, col, 14, width)
                    if rect_size > 0:
                        rect_count += 1
                        # Mark all cells of this rectangle as visited
                        # 重新掃描寬高以標記 visited (簡單處理)
                        w_temp = 0
                        while col + w_temp < width and level_2d[row][col + w_temp] == 'T': w_temp += 1
                        h_temp = 0
                        while row + h_temp < 14 and level_2d[row + h_temp][col] == 'T': h_temp += 1
                        
                        for r in range(row, row + h_temp):
                            for c in range(col, col + w_temp):
                                visited.add((r, c))
        
        return rect_count

# ==========================================
# 4. 主程式執行
# ==========================================
if __name__ == "__main__":
    print("--- 1. Generating X-Rectangles ---")
    
    # 初始化並執行填塞
    injector = RectInjector(FULL_LEVEL_STR_WITH_PATHS)
    injector.max_fill()
    
    # 存檔
    output_filename = "add_rect.txt"
    injector.save_to_file(output_filename)
    
    # 讀取檔案並驗證
    print("\n--- 2. Validating Result ---")
    with open(output_filename, "r") as f:
        content_lines = [line.strip() for line in f.readlines() if line.strip()]
        flattened_content = "".join(content_lines)
    
    validator = RectValidator()
    detected_count = validator.count_rects(flattened_content)
    
    print(f"🔍 Validated Rect Count from file: {detected_count}")