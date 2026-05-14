import sys
import os
import glob
import numpy as np

# --- 輔助類別：用於同時輸出到螢幕與檔案 ---
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w', encoding='utf-8')

    def log(self, message):
        print(message)
        self.log_file.write(message + '\n')

    def close(self):
        self.log_file.close()

class LevelIntegrityChecker:
    def __init__(self):
        self.SYMBOLS = {
            'PIPE_TOP_L': '<',
            'PIPE_TOP_R': '>',
            'PIPE_BODY_L': '[',
            'PIPE_BODY_R': ']',
            'GROUND': ['X', '#', 'S', 'Q', '?', 'o', 'E'], 
            'RECT': 'T'
        }

    def _analyze_rectangle(self, level_2d, start_row, start_col, rows):
        """
        分析矩形結構。
        支援中空矩形：只驗證邊框完整性，內部內容不限。
        回傳: (is_valid, is_broken, width, height)
        """
        current_line = level_2d[start_row]
        
        # 1. 探測寬度 (以頂邊為準)
        w = 0
        while start_col + w < len(current_line) and current_line[start_col + w] == self.SYMBOLS['RECT']:
            w += 1
        
        # 2. 探測高度 (以左邊為準)
        h = 0
        while start_row + h < rows and start_col < len(level_2d[start_row + h]) and level_2d[start_row + h][start_col] == self.SYMBOLS['RECT']:
            h += 1

        # 矩形至少要 2x2 才能構成幾何邊框
        if w < 2 or h < 2:
            return False, False, 0, 0

        is_broken = False

        # 3. 檢查邊框完整性 (Frame Check)
        for c in range(start_col, start_col + w):
            if level_2d[start_row][c] != self.SYMBOLS['RECT']: is_broken = True
            if (start_row + h - 1) >= rows or c >= len(level_2d[start_row + h - 1]) or level_2d[start_row + h - 1][c] != self.SYMBOLS['RECT']:
                is_broken = True
        
        for r in range(start_row, start_row + h):
            if start_col >= len(level_2d[r]) or level_2d[r][start_col] != self.SYMBOLS['RECT']:
                is_broken = True
            if (start_col + w - 1) >= len(level_2d[r]) or level_2d[r][start_col + w - 1] != self.SYMBOLS['RECT']:
                is_broken = True

        # 4. 排除檢查：確保這不是更大結構的一部分
        if not is_broken:
            if start_col + w < len(current_line) and current_line[start_col + w] == self.SYMBOLS['RECT']:
                return False, False, w, h
            if start_row + h < rows and start_col < len(level_2d[start_row + h]) and level_2d[start_row + h][start_col] == self.SYMBOLS['RECT']:
                return False, False, w, h

        return (not is_broken), is_broken, w, h

    def check_integrity(self, file_path):
        """核心檢測函數"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()][:14]
        except:
            return 0, 0, 0, 0

        if len(lines) < 14: return 0, 0, 0, 0
        
        rows = 14
        max_width = max(len(l) for l in lines)
        pipe_visited = set()
        v_pipes, b_pipes = 0, 0
        
        # --- 水管檢測 ---
        for r in range(rows):
            line = lines[r]
            for c in range(len(line)):
                char = line[c]
                if char in [self.SYMBOLS['PIPE_TOP_L'], self.SYMBOLS['PIPE_TOP_R'], 
                            self.SYMBOLS['PIPE_BODY_L'], self.SYMBOLS['PIPE_BODY_R']] and (r, c) not in pipe_visited:
                    
                    is_p_broken = False
                    curr_c = c if char in [self.SYMBOLS['PIPE_TOP_L'], self.SYMBOLS['PIPE_BODY_L']] else c - 1
                    if curr_c < 0 or curr_c + 1 >= len(line):
                        pipe_visited.add((r, c)); continue  # 邊界截斷，不計入

                    t_r = r
                    while t_r > 0 and curr_c < len(lines[t_r-1]) and lines[t_r-1][curr_c] in [self.SYMBOLS['PIPE_TOP_L'], self.SYMBOLS['PIPE_BODY_L']]:
                        t_r -= 1
                    b_r = r
                    while b_r < rows - 1 and curr_c < len(lines[b_r+1]) and lines[b_r+1][curr_c] in [self.SYMBOLS['PIPE_TOP_L'], self.SYMBOLS['PIPE_BODY_L']]:
                        b_r += 1

                    has_top, has_bot = False, False
                    for scan_r in range(t_r, b_r + 1):
                        s_line = lines[scan_r]
                        l_c = s_line[curr_c] if curr_c < len(s_line) else ' '
                        r_c = s_line[curr_c+1] if curr_c+1 < len(s_line) else ' '
                        pipe_visited.add((scan_r, curr_c)); pipe_visited.add((scan_r, curr_c+1))
                        
                        is_op = (l_c == self.SYMBOLS['PIPE_TOP_L'] and r_c == self.SYMBOLS['PIPE_TOP_R'])
                        is_bd = (l_c == self.SYMBOLS['PIPE_BODY_L'] and r_c == self.SYMBOLS['PIPE_BODY_R'])
                        if not (is_op or is_bd): is_p_broken = True
                        if is_op:
                            if scan_r == t_r: has_top = True
                            if scan_r == b_r: has_bot = True
                            if scan_r != t_r and scan_r != b_r: is_p_broken = True

                    if has_top and not has_bot:
                        if b_r < rows - 1 and (curr_c+1 >= len(lines[b_r+1]) or lines[b_r+1][curr_c] not in self.SYMBOLS['GROUND']): is_p_broken = True
                    elif has_bot and not has_top:
                        if t_r > 0 and (curr_c+1 >= len(lines[t_r-1]) or lines[t_r-1][curr_c] not in self.SYMBOLS['GROUND']): is_p_broken = True
                    elif not (has_top or has_bot): is_p_broken = True

                    if is_p_broken: b_pipes += 1
                    else: v_pipes += 1

        # --- 矩形檢測 (修復區域全覆蓋標記) ---
        v_rects, b_rects = 0, 0
        rect_visited = set()
        for r in range(rows):
            line = lines[r]
            for c in range(len(line)):
                if (r, c) not in rect_visited and line[c] == self.SYMBOLS['RECT']:
                    is_v, is_b, w, h = self._analyze_rectangle(lines, r, c, rows)
                    
                    if is_v or is_b:
                        at_boundary = (c == 0 or c + w >= max_width)
                        if not at_boundary:
                            if is_v: v_rects += 1
                            else: b_rects += 1
                        for i in range(r, r + h):
                            for j in range(c, c + w):
                                rect_visited.add((i, j))
                    else:
                        rect_visited.add((r, c))

        return v_pipes, b_pipes, v_rects, b_rects

def batch_process_folder(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    prefix = "multi_iter_test_"
    report_base = folder_name.replace(prefix, "") if folder_name.startswith(prefix) else folder_name
    report_filename = f"{report_base}_綜合完整度報告.txt"
    report_path = os.path.join(folder_path, report_filename)

    logger = DualLogger(report_path)
    checker = LevelIntegrityChecker()

    logger.log(f"📂 正在掃描資料夾: {folder_path}")
    
    # === [核心修改] 只搜尋生成的地圖檔案，排除報告類檔案 ===
    # 使用通配符只匹配特定的生成檔案格式
    search_pattern = os.path.join(folder_path, "moe_generated_level_*.txt")
    files = glob.glob(search_pattern)
    
    # 過濾掉可能存在的報告檔案（雙重保險）
    files = [f for f in files if os.path.basename(f) != report_filename and "report" not in os.path.basename(f)]
    
    if not files:
        logger.log(f"❌ 找不到符合格式 'moe_generated_level_*.txt' 的檔案")
        logger.close()
        return

    g_v_p, g_b_p, g_v_r, g_b_r = 0, 0, 0, 0

    logger.log("\n" + "-" * 110)
    logger.log(f"{'檔案名稱':<35} | {'完好水管':<8} | {'破損水管':<8} | {'完好矩形':<8} | {'破損矩形':<8}")
    logger.log("-" * 110)

    # 排序檔案，讓輸出更有條理
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]) if os.path.basename(x).split('_')[-1].split('.')[0].isdigit() else 0)

    for f_path in files:
        vp, bp, vr, br = checker.check_integrity(f_path)
        g_v_p += vp; g_b_p += bp; g_v_r += vr; g_b_r += br
        mark = "❌" if (bp > 0 or br > 0) else " "
        logger.log(f"{os.path.basename(f_path):<35} | {vp:<12} | {bp:<12} | {vr:<12} | {br:<12} {mark}")

    total_p = g_v_p + g_b_p
    total_r = g_v_r + g_b_r
    p_rate = (g_v_p / total_p * 100) if total_p > 0 else 0.0
    r_rate = (g_v_r / total_r * 100) if total_r > 0 else 0.0

    logger.log("-" * 110)
    logger.log("\n📊 === 最終統計報告 (Final Summary) ===")
    logger.log(f"總檢測檔案數: {len(files)}")
    logger.log(f"-----------------------------------")
    logger.log(f"🧪 水管 (Pipes): 總數 {total_p}, 完好 {g_v_p}, 破損 {g_b_p}")
    logger.log(f"📈 水管完整率: {p_rate:.2f}%")
    logger.log(f"-----------------------------------")
    logger.log(f"🧱 矩形 (Rects): 總數 {total_r}, 完好 {g_v_r}, 破損 {g_b_r}")
    logger.log(f"📉 矩形完整率: {r_rate:.2f}%")
    logger.log("===================================\n")
    logger.close()

if __name__ == "__main__":
    TARGET_FOLDER = r"C:\PYTHONPJ\AI\mario-gpt-moe-mlp-2DROPE\mario_gpt\multi_iter_test"
    if os.path.exists(TARGET_FOLDER):
        batch_process_folder(TARGET_FOLDER)
    else:
        print(f"❌ 路徑不存在: {TARGET_FOLDER}")