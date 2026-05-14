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
    y_control = y1 - (jump_rise * 1.5) 
    if y_control > y2 - 1:
        y_control = y2 - 1
    if y_control < 0: y_control = 0
    
    for xi in range(x1 + 1, x2):
        t = (xi - x1) / float(x2 - x1)
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

    print(f">>> Start Simulating from {curr_pos}")

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

            height_diff = cy - ty 
            if height_diff >= 6: continue 
            
            state[2] = i + 1 
            
            # 1. 基礎路徑碰撞檢查
            if get_valid_l_path(grid, cx, cy, tx, ty):
                token, is_jump, rise = "M", False, 0
                
                # 2. 行為判定與算法選擇
                if cy < ty: # Fall
                    is_jump = (dx >= 3)
                    rise = 1 if is_jump else 0
                    token = "j" if is_jump else "M"
                    traj_func = calculate_parabola_points
                elif cy > ty: # Climb
                    is_jump = True
                    token, rise = ("J", 4) if abs(dy) >= 4 else ("j", 2)
                    traj_func = calculate_parabola_points_2
                else: # Flat
                    if dx >= 5: token, is_jump, rise = "J", True, 4
                    elif 2 <= dx <= 4: token, is_jump, rise = "j", True, 2
                    elif dx == 1: token, is_jump = "M", False
                    traj_func = calculate_parabola_points

                # 3. [核心修改]：檢查目標點是否為錢幣 'o'
                # 如果是錢幣，則將該點的 Token 標記為 'C'
                final_token = token
                if grid[ty, tx] == 'o':
                    final_token = 'C'

                # 4. 生成路徑點
                new_path_map = copy.deepcopy(path_map)
                if is_jump:
                    traj = traj_func(cx, cy, tx, ty, rise, token)
                    new_path_map[(cx, cy)] = token
                    for px, py, pt in traj: 
                        # 拋物線過程中如果經過錢幣也可以考慮標註，但這裡優先標註落腳點
                        new_path_map[(px, py)] = pt
                
                new_pos = (tx, ty)
                new_path_map[new_pos] = final_token # 使用判定後的 token (可能是 M, J, j 或 C)
                
                path_map = new_path_map
                new_targets = get_sorted_targets(tx, ty, failed_spots)
                history_stack.append([new_pos, new_targets, 0, copy.deepcopy(path_map)])
                found_move = True
                break

        if not found_move:
            backtrack_count += 1
            bad_state = history_stack.pop()
            failed_spots.add(bad_state[0]) 
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
    final_path_map = simulate_gameplay(grid)
    
    combined_grid = np.copy(grid)
    for (x, y), token in final_path_map.items():
        if 0 <= y < H and 0 <= x < W:
            # 覆蓋邏輯：如果是空氣或錢幣，直接填入動作 TOKEN
            # 如果原本是實體但路徑標註為 M/J/j/C（通常是落腳點），也進行覆蓋
            if not is_solid(combined_grid[y, x]) or combined_grid[y, x] == 'o':
                combined_grid[y, x] = token
            elif token in {'M', 'J', 'j', 'C'}:
                combined_grid[y, x] = token

    with open("play.txt", "w", encoding="utf-8") as f:
        for row in combined_grid:
            f.write("".join(row) + "\n")
    print(f"Results saved to play.txt. Total steps: {len(final_path_map)}")

if __name__ == "__main__":
    main()