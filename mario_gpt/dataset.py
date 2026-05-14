from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS

DEFAULT_MODEL = "distilgpt2"
DEBUG = False

def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


def flip_and_transpose(arr: np.array, flip_first: bool = False):
    if arr.shape[-1] > 1:
        if flip_first:
            return np.flip(arr, -1).transpose()
        return np.flip(arr.transpose(), -1)
    return arr


def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]


def characterize(str_lists):
    return [list(s) for s in str_lists]

def build_positions_2d(context_len: int, height: int = 14, device="cpu"):
    t = torch.arange(context_len, device=device)
    x = t // height
    y = (height - 1) - (t % height)  # bottom->top
    return torch.stack([x, y], dim=-1).to(torch.float32)  # (T,2)

class MarioDataset(Dataset):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        level_string: Optional[str] = None,
        context_len: int = 700,
        height: int = 14,
        remove_start_end_tokens: bool = False,
        sample_all_indices: bool = False, # True
    ):
        # 1. 載入關卡字串 (Level String Loading)
        if level_string is None:
            print(
                "--------- No level string specified, using default string FULL_LEVEL_STR_WITH_PATHS..."
            )
            level_string = FULL_LEVEL_STR_WITH_PATHS
        elif ".txt" in level_string:
            print(f"-------- Loading level string from file...{level_string}")
            with open(level_string, "r") as file:
                level_string = file.read()

        # [新增功能 1] 保存原始字串 (來自代碼段 2)
        # 用於後續可能的 Grid 還原檢查或 Debug
        self.raw_level_string = level_string

        self.character_set = set(level_string)

        print(f"==================================")
        print(f"Character set: {self.character_set}")

        if "\n" in self.character_set:
            self.character_set.remove("\n")
        self.vocab_size = len(self.character_set)
        self.sample_all_indices = sample_all_indices

        def get_training_corpus():
            yield list(level_string)

        # 2. Tokenizer 設定 (Tokenizer Setup)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

        self.tokenizer = tokenizer
        if getattr(tokenizer, "train_new_from_iterator", None) is not None:
            print('-----------tokenizer: train_new_from_iterator')
            self.tokenizer = self.tokenizer.train_new_from_iterator(
                get_training_corpus(), 52000
            )
        elif getattr(tokenizer, "train_from_iterator", None) is not None:
            print('-----------tokenizer: train_from_iterator')
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
            self.tokenizer = self.tokenizer.train_new_from_iterator(
                get_training_corpus(), self.vocab_size
            )
        self.context_len = context_len
        self.height = height

        # 3. 轉換為 Tensor (Convert to Tensor)
        x, self.str_arr = self.convert_level_to_tensor(level_string.split("\n"))
        self.input_ids = x["input_ids"].squeeze()
        self.attention_masks = x["attention_mask"].squeeze()
        if remove_start_end_tokens:
            self.input_ids = self.input_ids[1:-1]
            self.attention_masks = self.attention_masks[1:-1]

        # ======================================================
        # [保留功能] 生成 2D 位置編碼 (來自代碼段 1 - 2D-RoPE 核心)
        # Generate 2D position encoding for the entire sequence
        # ======================================================
        # 
        # 注意：這裡呼叫的是外部定義的 build_positions_2d 函式
        self.positions_2d_full = build_positions_2d(self.context_len, self.height)
        
        if DEBUG:
            print(f'dataset:self.positions_2d_full:{self.positions_2d_full.shape}')

        # 4. 生成索引與統計 (Indices & Stats)
        self.indices = self.generate_indices()

        self.unique_tokens, self.unique_counts = self.input_ids.unique(
            return_counts=True
        )
        self.weighted_unique_counts = (
            1.0 / self.unique_counts / torch.sum(self.unique_counts)
        )

        self.token_dict = {}
        string_tokens = list(self.tokenizer.decode(self.unique_tokens))
        for int_token, string_token in zip(self.unique_tokens, string_tokens):
            self.token_dict[string_token] = int_token

        # ---------------------------------------------------
        # [新增功能 2] 預先快取關鍵 Token ID (來自代碼段 2)
        # 用於 __getitem__ 中的 process_window 快速修復水管
        # ---------------------------------------------------
        try:
            # 嘗試從 token_dict 獲取，若無則即時 encode
            self.id_dash = self.token_dict.get("-", self.tokenizer.encode("-", add_special_tokens=False)[0])
            self.id_pipe_l_body = self.token_dict.get("[", self.tokenizer.encode("[", add_special_tokens=False)[0])
            self.id_pipe_r_body = self.token_dict.get("]", self.tokenizer.encode("]", add_special_tokens=False)[0])
            self.id_pipe_l_top = self.token_dict.get("<", self.tokenizer.encode("<", add_special_tokens=False)[0])
            self.id_pipe_r_top = self.token_dict.get(">", self.tokenizer.encode(">", add_special_tokens=False)[0])
        except Exception as e:
            if DEBUG:
                print(f"Warning: Failed to cache specific token IDs: {e}")
            # Fallback 防止某些特殊 tokenizer 狀況
            self.id_dash = 0
            self.id_pipe_l_body, self.id_pipe_r_body = -1, -1
            self.id_pipe_l_top, self.id_pipe_r_top = -1, -1

    def convert_level_to_tensor(self, level: List[str]):
        # 確保有這些 helper function: characterize, flip_and_transpose, join_list_of_list
        ft = np.array(characterize(level))
        str_arr = flip_and_transpose(ft)
        str_arr = "".join(join_list_of_list(str_arr))

        x = self.tokenizer(str_arr, return_tensors="pt")
        return x, str_arr


    def convert_level_to_tensor(self, level: List[str]):
        str_arr = flip_and_transpose(np.array(characterize(level)))
        str_arr = "".join(join_list_of_list(str_arr))

        x = self.tokenizer(str_arr, return_tensors="pt")
        return x, str_arr

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        # ---------------------------------------------------
        # 情況 A: 批次處理 (List, Tuple, Tensor, Array)
        # ---------------------------------------------------
        if isinstance(idx, (list, tuple, torch.Tensor, np.ndarray)):
            # 1. 確保 idx 是標準列表格式
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            elif isinstance(idx, np.ndarray):
                idx = idx.tolist()

            # 2. 呼叫修復邏輯 (來自第二段代碼)
            # 對每個索引跑一次 process_window，取得 (input_ids, masks)
            # 注意：這裡使用 process_window 取代了原本的 self.input_ids[indices]
            results = [self.process_window(int(i)) for i in idx]
            
            # 3. 解包並堆疊 Tensor
            batch_ids, batch_masks = zip(*results)
            input_ids = torch.stack(batch_ids)          # (Batch_Size, T)
            attention_masks = torch.stack(batch_masks)  # (Batch_Size, T)

            # 4. 補上 2D 座標 (來自第一段代碼)
            # 因為每個視窗大小固定 (14x50)，座標對於每個樣本都是一樣的
            # 我們將 (T, 2) 擴展並複製為 (Batch_Size, T, 2)
            pos2d = self.positions_2d_full.unsqueeze(0).repeat(input_ids.shape[0], 1, 1)

            return input_ids, attention_masks, pos2d

        # ---------------------------------------------------
        # 情況 B: 單筆處理 (Integer)
        # ---------------------------------------------------
        else:
            # 1. 呼叫修復邏輯
            input_ids, attention_masks = self.process_window(int(idx)) # 回傳 (T,)

            # 2. 補上 2D 座標
            # 單筆資料直接回傳預存的座標表即可 (T, 2)
            pos2d = self.positions_2d_full 

            return input_ids, attention_masks, pos2d


    def process_window(self, idx):
        """
        處理單一視窗：切片 -> 修復水管 -> 回傳 Tensor
        """
        indices = self.indices[idx]
        
        # 必須 clone，否則修改會影響到原始 self.input_ids
        x = self.input_ids[indices].clone()
        m = self.attention_masks[indices].clone()

        # 如果視窗長度不足或是沒有高度資訊，直接回傳
        T = x.shape[0]
        H = self.height
        if T < H: 
            return x, m

        W = T // H
        
        # 這裡我們直接操作 Tensor ID，比轉回字串再轉回來快很多
        # 第一列 (First Column) 的 Indices: 0 到 H
        first_col_ids = x[0:H]
        # 最後一列 (Last Column) 的 Indices: (W-1)*H 到 W*H
        last_col_ids = x[(W-1)*H : W*H]

        # --- 修復開頭 (First Column) ---
        # 檢查是否有孤單的右半邊 ( ] 或 > )
        # 使用 torch.isin 或是直接比較
        has_broken_r = (first_col_ids == self.id_pipe_r_body).any() or \
                       (first_col_ids == self.id_pipe_r_top).any()
        
        if has_broken_r:
            # 將該列所有的 ] 和 > 替換成 -
            mask_r = (first_col_ids == self.id_pipe_r_body) | (first_col_ids == self.id_pipe_r_top)
            # 在 Tensor 上直接替換值
            x[0:H][mask_r] = self.id_dash

        # --- 修復結尾 (Last Column) ---
        # 檢查是否有孤單的左半邊 ( [ 或 < )
        has_broken_l = (last_col_ids == self.id_pipe_l_body).any() or \
                       (last_col_ids == self.id_pipe_l_top).any()
        
        if has_broken_l:
            # 將該列所有的 [ 和 < 替換成 -
            mask_l = (last_col_ids == self.id_pipe_l_body) | (last_col_ids == self.id_pipe_l_top)
            base = (W-1) * H
            # 在 Tensor 上直接替換值
            x[base : base+H][mask_l] = self.id_dash

        return x, m

    def generate_indices(self):
        out = []
        for idx in range(self.input_ids.shape[0] - self.context_len):
            if idx % self.height == 0 or self.sample_all_indices:
                arange = torch.arange(idx, idx + self.context_len)
                out.append(arange)
        return torch.stack(out)

    def sample_indices(self, batch_size):
        out = []
        for _ in range(batch_size):
            start_idx = np.random.randint(0, self.__len__() - self.context_len)
            indices = torch.arange(start_idx, start_idx + self.context_len)
            out.append(indices)
        return torch.stack(out)

    def __str__(self):
        # 修正後的 __str__，避免使用不存在的 self.x
        if hasattr(self, 'input_ids'):
            # 為了列印方便，只取前 context_len 個 token 來展示
            display_ids = self.input_ids[:self.context_len].cpu().unsqueeze(0)
            str_list = characterize(self.tokenizer.batch_decode(display_ids))
            string = "\n".join(
                join_list_of_list(flip_and_transpose(np.array(str_list), True))
            )
            return string
        return "MarioDataset"