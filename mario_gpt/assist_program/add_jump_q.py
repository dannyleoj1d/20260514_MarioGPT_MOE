#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

def is_solid(ch: str) -> bool:
    """定義哪些 Token 會阻擋跳躍路徑（不含目標物）"""
    return ch in {"X", "S", "Q", "B", "b", "<", ">", "[", "]", "T", "E", "B"}

def process_jump_logic(input_file="play.txt", output_file="play_jumped.txt"):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [list(line.strip()) for line in f.readlines() if line.strip()]
        
        grid = np.array(lines)
        H, W = grid.shape
        
        # 掃描整個地圖尋找 M
        for y in range(H):
            for x in range(W):
                if grid[y, x] == 'M':
                    max_scan_dist = 4 
                    targets_found = []  # 儲存 (y, type)
                    
                    # 完整掃描上方 4 格
                    for dist in range(1, max_scan_dist + 1):
                        check_y = y - dist
                        if check_y < 0:
                            break
                            
                        curr_token = grid[check_y, x]
                        
                        if curr_token in {'?', 'o'}:
                            targets_found.append((check_y, curr_token))
                            # 發現目標後繼續往上掃描，看有沒有第二層金幣
                        elif is_solid(curr_token):
                            # 撞到實體牆壁，上方再有金幣也吃不到，停止掃描
                            break
                    
                    if targets_found:
                        # 1. 將起始點 M 轉換為 A (發起跳躍動作)
                        grid[y, x] = 'A'
                        
                        # 2. 獲取最高目標的 y 座標，用來建立路徑連線
                        highest_target_y = min([t[0] for t in targets_found])
                        
                        # 3. 處理所有發現的目標 (如果是金幣 o 就換成 C)
                        for ty, t_type in targets_found:
                            if t_type == 'o':
                                grid[ty, x] = 'C'
                        
                        # 4. 建立垂直路徑連線 (將 M 到最高目標之間的所有空氣填為 J)
                        for fill_y in range(highest_target_y + 1, y):
                            if grid[fill_y, x] == '-':
                                grid[fill_y, x] = 'J'
        
        # 存檔
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in grid:
                f.write("".join(row) + "\n")
                
        print(f"成功！已處理多層跳躍邏輯，結果存至: {output_file}")

    except Exception as e:
        print(f"處理失敗: {e}")

if __name__ == "__main__":
    process_jump_logic()