#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import random
import math
import copy
from typing import List, Tuple, Dict, Optional, Set

# ==========================================
# 1. Import Source Map
# ==========================================
try:
    from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS
except ImportError:
    print("[WARNING] Cannot import 'FULL_LEVEL_STR_WITH_PATHS'. Using dummy map.")
    FULL_LEVEL_STR_WITH_PATHS = "XXXXXXXXXXXXXXXXXXXX\n--------------------\nXXXXXXXXXXXXXXXXXXXX"

# ==========================================
# 2. Helper Functions
# ==========================================

def is_solid(ch: str) -> bool:
    return ch in {"X", "S", "?", "Q", "B", "b", "<", ">", "[", "]", "T"}

def str_to_grid(level_str: str) -> np.ndarray:
    lines = [line.strip() for line in level_str.strip().split('\n') if line.strip()]
    if not lines:
        raise ValueError("Level string is empty!")
    width = max(len(line) for line in lines)
    padded_lines = [line.ljust(width, '-') for line in lines]
    return np.array([list(line) for line in padded_lines])

def get_valid_l_path(grid: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    H, W = grid.shape
    # 路線 1: 先 X 後 Y
    p1_ok = True
    step_x = 1 if x2 >= x1 else -1
    for x in range(x1, x2 + step_x, step_x):
        if is_solid(grid[y1, x]): p1_ok = False; break
    if p1_ok:
        step_y = 1 if y2 >= y1 else -1
        for y in range(y1, y2 + step_y, step_y):
            if is_solid(grid[y, x2]): p1_ok = False; break
    if p1_ok: return True

    # 路線 2: 先 Y 後 X
    p2_ok = True
    step_y = 1 if y2 >= y1 else -1
    for y in range(y1, y2 + step_y, step_y):
        if is_solid(grid[y, x1]): p2_ok = False; break
    if p2_ok:
        step_x = 1 if x2 >= x1 else -1
        for x in range(x1, x2 + step_x, step_x):
            if is_solid(grid[y2, x]): p2_ok = False; break
    return p2_ok

def calculate_parabola_points(x1: int, y1: int, x2: int, y2: int, jump_rise: int, token: str) -> List[Tuple[int, int, str]]:
    points = []
    peak_y = y1 - jump_rise
    target_min_y = min(y1, y2)
    if peak_y > target_min_y - 1: peak_y = target_min_y - 2
    if peak_y < 0: peak_y = 0
    for xi in range(x1 + 1, x2):
        t = (xi - x1) / float(x2 - x1)
        y_linear = y1 + (y2 - y1) * t
        arch = 4 * t * (1 - t)
        y_mid_linear = (y1 + y2) / 2.0
        lift = y_mid_linear - peak_y
        yi = y_linear - (arch * lift)
        points.append((xi, int(round(yi)), token))
    return points

# ==========================================
# 3. Core Logic
# ==========================================

def get_all_standable_spots(grid: np.ndarray) -> List[Tuple[int, int]]:
    H, W = grid.shape
    spots = []
    for y in range(H - 1):
        for x in range(W):
            if is_solid(grid[y+1, x]) and not is_solid(grid[y, x]):
                spots.append((x, y))
    return spots

def simulate_gameplay(grid: np.ndarray) -> Dict[Tuple[int, int], str]:
    H, W = grid.shape
    all_spots = get_all_standable_spots(grid)
    
    start_candidates = [pos for pos in all_spots if pos[0] == 0]
    if not start_candidates: return {}
    curr_pos = random.choice(start_candidates)
    
    # 初始化
    path_map = {curr_pos: "M"}
    
    # [新增] 儲存已經被證實為「死胡同」或「不通」的座標
    failed_spots: Set[Tuple[int, int]] = set()
    
    # 搜尋記錄 (用於回溯)
    # 格式: [ [當前座標, 候選目標列表, 已經試過的索引, 路徑圖快照] ]
    history_stack = []
    
    def get_sorted_targets(cx, cy, current_failed_set: Set[Tuple[int, int]]):
        # 取得前方目標並按歐幾里得距離排序，同時過濾掉已經失敗的點
        targets = [pos for pos in all_spots if pos[0] > cx and pos not in current_failed_set]
        targets.sort(key=lambda pos: math.sqrt((pos[0]-cx)**2 + (pos[1]-cy)**2))
        return targets

    current_targets = get_sorted_targets(curr_pos[0], curr_pos[1], failed_spots)
    history_stack.append([curr_pos, current_targets, 0, copy.deepcopy(path_map)])
    
    backtrack_count = 0
    MAX_BACKTRACK = 1000 # 搜尋地獄上限

    print(f">>> Start Simulating from {curr_pos}, Ground: '{grid[curr_pos[1]+1, curr_pos[0]]}'")

    while history_stack:
        state = history_stack[-1]
        (cx, cy) = state[0]
        targets = state[1]
        t_idx = state[2]
        
        # 檢查是否接近終點
        if cx >= W - 5:
            print(f"!!! Goal Reached at x={cx} !!!")
            return path_map

        found_move = False
        # 從上一次嘗試的索引開始繼續往下找
        for i in range(t_idx, len(targets)):
            tx, ty = targets[i]
            
            # 再次檢查該目標是否在全域失敗清單中 (可能在進入此層後被其他分支標記為失敗)
            if (tx, ty) in failed_spots:
                continue

            dist = math.sqrt((tx - cx)**2 + (ty - cy)**2)
            if dist > 14: continue 
            
            # 更新當前層嘗試的索引
            state[2] = i + 1
            ground_char = grid[ty + 1, tx]

            # 雙路徑判斷
            if get_valid_l_path(grid, cx, cy, tx, ty):
                # 判定動作分歧
                dx = tx - cx
                dy = cy - ty 
                
                token, is_jump, rise = "M", False, 0
                if cy < ty: # Fall
                    token = "M"; is_jump = (dx >= 2) # 高於目標且寬度大於等於2視為跳躍
                    if is_jump: rise = 1
                elif cy > ty: # Climb
                    is_jump = True
                    token, rise = ("J", 4) if dy >= 4 else ("j", 2)
                else: # Flat (高度相同)
                    if dx >= 5: token, is_jump, rise = "J", True, 4
                    elif 2 <= dx <= 4: token, is_jump, rise = "j", True, 2
                    elif dx == 1: token, is_jump = "M", False

                # 更新快照
                new_path_map = copy.deepcopy(path_map)
                if is_jump:
                    traj = calculate_parabola_points(cx, cy, tx, ty, rise, token)
                    new_path_map[(cx, cy)] = token
                    for px, py, pt in traj: new_path_map[(px, py)] = pt
                
                new_pos = (tx, ty)
                new_path_map[new_pos] = "M"
                
                print(f"  [MOVE] ({cx},{cy}) -> ({tx},{ty}) | Ground: '{ground_char}' | Action: {token}")
                
                # 更新全局路徑並壓入堆疊進入下一層
                path_map = new_path_map
                new_targets = get_sorted_targets(tx, ty, failed_spots)
                history_stack.append([new_pos, new_targets, 0, copy.deepcopy(path_map)])
                found_move = True
                break
        
        # 如果當前點所有目標都失敗 -> 標記為不通並回溯
        if not found_move:
            backtrack_count += 1
            bad_state = history_stack.pop()
            bad_pos = bad_state[0]
            
            # [核心改動] 將此座標標記為「不通」，之後的路徑搜尋將不再考慮此點
            failed_spots.add(bad_pos)
            
            if not history_stack:
                print("[FAILURE] No path possible from origin after exploring all branches.")
                break
            
            # 偵測搜尋地獄
            if backtrack_count > MAX_BACKTRACK:
                print(f"\n[ALERT] SEARCH HELL AT x={bad_pos[0]}!")
                break

            # 恢復上一層狀態
            path_map = copy.deepcopy(history_stack[-1][3])
            # 更新上一層的目標清單，剔除剛被標記為失敗的點
            history_stack[-1][1] = [t for t in history_stack[-1][1] if t not in failed_spots]
            
            print(f"  [BACKTRACK] #{backtrack_count} | Spot {bad_pos} marked as 'FAILED' | Returning to {history_stack[-1][0]}")

    return path_map

# ==========================================
# 4. Main Program
# ==========================================

def main():
    print("Loading Level...")
    try:
        grid = str_to_grid(FULL_LEVEL_STR_WITH_PATHS)
    except Exception as e:
        print(f"[ERROR] {e}"); return

    H, W = grid.shape
    print(f"Map Size: {H}x{W}")
    
    final_path_map = simulate_gameplay(grid)
    
    combined_grid = np.copy(grid)
    for (x, y), token in final_path_map.items():
        if 0 <= y < H and 0 <= x < W:
            if not is_solid(combined_grid[y, x]):
                combined_grid[y, x] = token
            elif token in {'M', 'J', 'j'}:
                combined_grid[y, x] = token

    with open("play.txt", "w", encoding="utf-8") as f:
        for row in combined_grid:
            f.write("".join(row) + "\n")
    print(f"Results saved to play.txt. Total success steps: {len(final_path_map)}")

if __name__ == "__main__":
    main()