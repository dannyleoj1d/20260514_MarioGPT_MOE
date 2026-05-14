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
    FULL_LEVEL_STR_WITH_PATHS = "--------------------\n--------------------\nXXXXXXXXXXXXXXXXXXXX"

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
    p1_ok = True
    step_x = 1 if x2 >= x1 else -1
    for x in range(x1, x2 + step_x, step_x):
        if is_solid(grid[y1, x]): p1_ok = False; break
    if p1_ok:
        step_y = 1 if y2 >= y1 else -1
        for y in range(y1, y2 + step_y, step_y):
            if is_solid(grid[y, x2]): p1_ok = False; break
    if p1_ok: return True

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
    if jump_rise > 0:
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

def calculate_parabola_points_2(x1: int, y1: int, x2: int, y2: int, jump_rise: int, token: str) -> List[Tuple[int, int, str]]:
    points = []
    
    # 1. 鎖定最高點高度：嚴格由起點(y1)決定，不被 y2 拉高
    # 我們讓 y_control 稍微比目標 y1 高度多一點 offset，確保軌跡漂亮
    y_control = y1 - (jump_rise * 1.5) 
    
    # 安全檢查：至少要比目標 y2 高出一格，否則會直接撞在平台側邊
    if y_control > y2 - 1:
        y_control = y2 - 1
    if y_control < 0: y_control = 0

    # 2. 為了讓頂點靠近目標點，我們需要模擬非對稱的移動
    # 我們不再使用單純的貝茲公式，改用一個帶權重的 t 分佈
    # 或者更簡單地：手動分段或調整控制點 X 偏移
    
    for xi in range(x1 + 1, x2):
        t = (xi - x1) / float(x2 - x1)
        
        # 使用一個 Power Function 調整 t 的感官速度 (讓它後半段才上升到頂點)
        # bias_t 會讓頂點往 x2 靠攏
        bias_t = t ** 0.7  
        
        # 二次貝茲公式：P0 = y1, P1 = y_control, P2 = y2
        # 注意：我們使用原本的 t 來算 X，但用稍微偏移的比例來算 Y 軌跡
        yi = (1 - t)**2 * y1 + 2 * (1 - t) * t * y_control + t**2 * y2
        
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
    
    path_map = {curr_pos: "M"}
    failed_spots: Set[Tuple[int, int]] = set()
    history_stack = [] 
    
    def get_sorted_targets(cx, cy, current_failed_set: Set[Tuple[int, int]]):
        targets = [pos for pos in all_spots if pos[0] > cx and pos not in current_failed_set]
        targets.sort(key=lambda pos: math.sqrt((pos[0]-cx)**2 + (pos[1]-cy)**2))
        return targets

    current_targets = get_sorted_targets(curr_pos[0], curr_pos[1], failed_spots)
    history_stack.append([curr_pos, current_targets, 0, copy.deepcopy(path_map)])
    
    backtrack_count = 0
    MAX_BACKTRACK = 2000 

    print(f">>> Start Simulating from {curr_pos}, Ground: '{grid[curr_pos[1]+1, curr_pos[0]]}'")

    while history_stack:
        state = history_stack[-1]
        (cx, cy) = state[0]
        targets = state[1]
        t_idx = state[2]
        
        if cx >= W - 5:
            print(f"!!! Goal Reached at x={cx} !!!")
            return path_map

        found_move = False
        for i in range(t_idx, len(targets)):
            tx, ty = targets[i]
            if (tx, ty) in failed_spots: continue

            dx, dy = tx - cx, ty - cy
            dist = math.sqrt(dx**2 + dy**2)

            if dist > 14: continue

            # 高度限制檢查
            height_diff = cy - ty 
            if height_diff >= 6:
                continue 
            
            state[2] = i + 1 
            
            # 1. 基礎路徑碰撞檢查
            if get_valid_l_path(grid, cx, cy, tx, ty):
                ground_char = grid[ty + 1, tx]
                token, is_jump, rise = "M", False, 0
                
                # 2. 行為判定與算法引用選擇
                if cy < ty: # 往下掉 (Fall)
                    token = "M"
                    is_jump = (dx >= 3)
                    rise = 1 if is_jump else 0
                    if is_jump: token = "j"
                    # 下墜使用對稱算法
                    traj_func = calculate_parabola_points
                    
                elif cy > ty: # 往上跳 (Climb)
                    is_jump = True
                    token, rise = ("J", 4) if abs(dy) >= 4 else ("j", 2)
                    # [核心修改]: 當 PLAYER 比目標低 (cy > ty)，引用非對稱算法
                    traj_func = calculate_parabola_points_2
                    
                else: # 平地 (Flat)
                    if dx >= 5: token, is_jump, rise = "J", True, 4
                    elif 2 <= dx <= 4: token, is_jump, rise = "j", True, 2
                    elif dx == 1: token, is_jump = "M", False
                    # 平地跳躍使用對稱算法
                    traj_func = calculate_parabola_points

                # 3. 生成路徑點
                new_path_map = copy.deepcopy(path_map)
                if is_jump:
                    # 動態引用選定的函式
                    traj = traj_func(cx, cy, tx, ty, rise, token)
                    new_path_map[(cx, cy)] = token
                    for px, py, pt in traj: new_path_map[(px, py)] = pt
                
                new_pos = (tx, ty)
                new_path_map[new_pos] = "M"
                
                path_map = new_path_map
                new_targets = get_sorted_targets(tx, ty, failed_spots)
                history_stack.append([new_pos, new_targets, 0, copy.deepcopy(path_map)])
                found_move = True
                break

        if not found_move:
            backtrack_count += 1
            bad_state = history_stack.pop()
            bad_pos = bad_state[0]
            failed_spots.add(bad_pos) 
            
            if not history_stack: break
            
            path_map = copy.deepcopy(history_stack[-1][3])
            history_stack[-1][1] = [t for t in history_stack[-1][1] if t not in failed_spots]
            
            if backtrack_count > MAX_BACKTRACK: break

    return path_map

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