#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def extract_path_only(input_file="play.txt", output_file="path_only.txt"):
    """
    讀取 play.txt，只保留 'M', 'J', 'j' 三種路徑 Token，
    其餘所有 Token（包含實體障礙物）全部轉換為 '-'。
    """
    path_tokens = {'M', 'J', 'j'}
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        for line in lines:
            # 逐字元檢查，如果是路徑則保留，否則變為空氣 '-'
            # 使用 strip() 處理換行，再加回換行符號
            new_line = "".join([ch if ch in path_tokens else "-" for ch in line.strip()])
            cleaned_lines.append(new_line + "\n")
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
            
        print(f"成功！路徑已提取至: {output_file}")
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {input_file}，請確認檔案是否存在。")

if __name__ == "__main__":
    extract_path_only()