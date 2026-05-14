# -*- coding: utf-8 -*-
"""
MarioGPT Single-Tile IG Heatmap (Standard Embedding L2 Norm)
- MoE 架構（非雙專家），舊生成順序：column-first, bottom-to-top
- 僅計算 Input Embedding 的貢獻度，移除 QKV Hook
- Flatten 2D level to 1D exactly like MarioDataset
- 滑動視窗對齊 column 底部 (y=H-1)
"""

import os
import sys
import random
import numpy as np
import torch
from torch import amp
from matplotlib.colors import ListedColormap, BoundaryNorm

# IG.py lives inside mario_gpt/, so we need to add the parent (structure-expert/) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mario_gpt import MarioLM
from captum.attr import IntegratedGradients

# ================= 設定區 =================
NUM_EXPERTS = 8
MOE_TOP_K   = 2
TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
CHECKPOINT_DIR = "Mario-GPT2-700-context-length/iteration_99999"
BASE_MODEL     = "random"

LEVEL_HEIGHT = 14
IG_STEPS     = 16                 # 8~32 suggested
INTERNAL_BS  = 8
USE_BF16     = True               # CUDA: bfloat16; otherwise float16
ENABLE_GC    = True               # gradient checkpointing

# ---- Target settings：每種類型分別指定 (y, x)（0-based，14xW grid）----
TARGETS = {
    "rect": {"y": 3,  "x": 43},   # 矩形結構目標座標
    "plat": {"y": 8,  "x": 42},   # 平台結構目標座標
    "pipe": {"y": 9,  "x": 45},   # 水管結構目標座標
}

# ---- Overlay settings ----
ANNOTATE_MODE = "all"             # 顯示所有格子的字元
MAX_LABELS    = 300
DRAW_GRID     = False

# ---- Custom color levels (value threshold, color) ----
COLOR_LEVELS = [
    (0.00001, "#FFFFFF"),
    (0.001,   "#99CCFF"),
    (0.01,    "#0066FF"),
    (0.05,    "#0033AA"),
    (0.10,    "#FF6600"),
    (1.0,     "#FF0000"),
]

# ========= Fixed level (請在此貼上你的關卡字串) =========
generated_rect = """----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
-----------------------------TTTTTTTTTTTTTTT--------------------------------------------------------
-----------------------------T-------------T--------------------------------------------------------
-----------------------------T-------------T--------------------------------------------------------
-----------------------------T-------------T--------------------------------------------------------
-----------------------------T-------------T--------------------------------------------------------
-----------------------------T-------------T--------------------------------------------------------
-----------------------------TTTTTTTTTTTTTTT--------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------"""

generated_plat = """----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------SSSSSSSS-----SSSSSSSS---------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------"""

generated_pipe = """----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
--------------------------------------------<>------------------------------------------------------
--------------------------------------------[]------------------------------------------------------
--------------------------------------------[]------------------------------------------------------
--------------------------------------------[]------------------------------------------------------
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"""

# ========= Utils =========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def characterize(str_lists):
    return [list(s) for s in str_lists]

def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]

def flip_and_transpose(arr: np.ndarray):
    """(H,W) -> (W,H)，column-first, bottom-to-top（舊架構：地板先生成）。"""
    if arr.shape[-1] > 1:
        return np.flip(arr, axis=0).transpose()
    return arr

def flatten_like_dataset(level_lines):
    """將關卡依照舊 MarioDataset 的 column-first, bottom-to-top 順序展平為一維。"""
    ft  = np.array(characterize(level_lines))  # (H,W)
    arr = flip_and_transpose(ft)               # (W,H)
    s   = "".join(join_list_of_list(arr))
    return s

def build_positions_2d(context_len: int, height: int = 14, device: str = "cpu"):
    """回傳 shape: (T,2)，每個 t 的 (x,y)，符合 column-first, bottom-to-top。"""
    t = torch.arange(context_len, device=device)
    x = t // height
    y = (height - 1) - (t % height)  # y=13 地板先, y=0 天空後
    return torch.stack([x, y], dim=-1).to(torch.float32)

def build_token_positions(win_ids, tok, start_idx: int, pos2d_np: np.ndarray) -> torch.Tensor:
    """為 window 的每個 token 指定對應字元的 2D 位置，回傳 shape: (1, L, 2)。"""
    positions = []
    char_offset = 0
    n_chars = len(pos2d_np)
    for tid in win_ids.tolist():
        decoded = tok.decode([int(tid)], clean_up_tokenization_spaces=False)
        global_idx = min(start_idx + char_offset, n_chars - 1)
        positions.append(pos2d_np[global_idx])
        char_offset += max(1, len(decoded))
    return torch.tensor(np.array(positions), dtype=torch.float32).unsqueeze(0)  # [1, L, 2]

# ===== Captum Helpers =====
def make_forward_func(hf_model, attention_mask_1L, target_token_id, position_2d=None):
    def forward_on_embeds(inputs_embeds: torch.Tensor):
        bs    = inputs_embeds.size(0)
        device = inputs_embeds.device

        attn = attention_mask_1L
        if attn is not None:
            if attn.dim() == 2 and attn.size(0) == 1:
                attn = attn.expand(bs, -1).contiguous()
            attn = attn.long()

        pos2d = position_2d
        if pos2d is not None:
            if pos2d.size(0) == 1 and bs > 1:
                pos2d = pos2d.expand(bs, -1, -1).contiguous()
            pos2d = pos2d.to(device=device, dtype=torch.float32)

        use_amp   = (device.type == "cuda")
        amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16

        with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = hf_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                position_2d=pos2d,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        return out.logits[:, -1, target_token_id]
    return forward_on_embeds

def ig_on_ids(hf_model, input_ids_1L, target_id, internal_bs, position_2d=None):
    """計算 Input Embedding 的 IG 貢獻度。"""
    device = next(hf_model.parameters()).device
    wte    = hf_model.transformer.wte

    with torch.no_grad():
        embeds = wte(input_ids_1L.to(device)).detach()   # [1, L, H]
    embeds.requires_grad_()

    attn_mask      = torch.ones_like(input_ids_1L, device=device)
    forward_logits = make_forward_func(hf_model, attn_mask, target_id, position_2d=position_2d)

    ig       = IntegratedGradients(forward_logits)
    baseline = torch.zeros_like(embeds)
    
    attrib = ig.attribute(
        inputs=embeds,
        baselines=baseline,
        n_steps=IG_STEPS,
        internal_batch_size=internal_bs,
        return_convergence_delta=False
    )  # [1, L, H]

    # 取 L2 norm 作為每個 token 的最終貢獻度分數
    token_attr = attrib.norm(p=2, dim=-1).detach().cpu().numpy()[0]  # [L]
    return token_attr

# ===== 主分析邏輯 =====
def single_tile_ig_heatmap_sliding(
    mario_lm,
    level_lines,
    target_y,
    target_x,
    kind="unknown",
    out_prefix="tile_ig_dswin"
):
    tok      = mario_lm.tokenizer
    hf_model = mario_lm.lm.eval()

    H = len(level_lines)
    if H == 0:
        raise ValueError("關卡資料為空！請確認 generated_level2 是否有填入內容。")
    assert H == LEVEL_HEIGHT, f"Expect 14 rows, got {H}"
    W = max(len(r) for r in level_lines)

    # 1) Flatten
    flat_text = flatten_like_dataset(level_lines)
    T         = len(flat_text)

    # 2) 建立 2D positions
    pos2d    = build_positions_2d(T, height=H, device="cpu")
    pos2d_np = pos2d.numpy().astype(int)
    xs       = pos2d_np[:, 0]
    ys       = pos2d_np[:, 1]

    # 3) 找到 target char 的 global index
    target_t_candidates = np.where((xs == target_x) & (ys == target_y))[0]
    if target_t_candidates.size == 0:
        raise ValueError(f"找不到 (y={target_y}, x={target_x}) 對應的 index。請確認座標是否超出範圍。")
    target_t = int(target_t_candidates[0])

    # 4) Tokenizer safety
    tgt_ids = tok.encode(flat_text[target_t], add_special_tokens=False)
    if len(tgt_ids) == 0:
        raise ValueError(f"Tokenizer cannot encode target char: {repr(flat_text[target_t])}")
    target_id = int(tgt_ids[0])

    # 5) 建立 sliding window
    max_ctx   = getattr(hf_model.config, "n_positions", 1024)
    start_idx = max(0, target_t - max_ctx)

    while start_idx < target_t and ys[start_idx] != (H - 1):
        start_idx += 1
    if start_idx >= target_t:
        start_idx = max(0, target_t - H)
        while start_idx < target_t and ys[start_idx] != (H - 1):
            start_idx += 1
        if start_idx >= target_t:
            start_idx = max(0, target_t - max_ctx)

    win_len = target_t - start_idx
    assert win_len > 0, "Empty window after alignment."

    if win_len > max_ctx:
        overflow = win_len - max_ctx
        start_idx = start_idx + overflow
        win_len   = target_t - start_idx

    # 6) Tokenize window 文字
    window_text = flat_text[start_idx:target_t]
    win_ids     = tok(window_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    # 6.5) 為每個 token 建立對應的 2D 位置
    window_pos2d = build_token_positions(win_ids, tok, start_idx, pos2d_np)  # [1, L, 2]

    # 7) Run IG
    print(f"\n[INFO] 開始計算 IG (Window Size: {win_len}, Tokens: {len(win_ids)})...")
    all_ids    = win_ids.unsqueeze(0)
    token_attr = ig_on_ids(hf_model, all_ids, target_id, INTERNAL_BS, position_2d=window_pos2d)

    # 8) Project back to 2D
    heat_embed = np.zeros((H, W), dtype=np.float64)

    tok_strings = [
        tok.decode([int(t)], clean_up_tokenization_spaces=False)
        for t in win_ids.tolist()
    ]

    running_t = start_idx
    for i, s in enumerate(tok_strings):
        if not s:
            continue
        denom = max(1, len(s))
        per_char_attr = float(token_attr[i]) / denom

        for _ch in s:
            if 0 <= running_t < T:
                x = xs[running_t]
                y = ys[running_t]
                if 0 <= y < H and 0 <= x < W:
                    heat_embed[y, x] += per_char_attr
            running_t += 1

    # 9) Visualization
    import matplotlib.pyplot as plt, matplotlib as mpl
    import matplotlib.patheffects as pe

    def _showable(ch: str) -> str:
        if ch == "\n": return "⏎"
        if ch == "\t": return "↹"
        if ch == " ":  return "·"
        return ch

    tgt_char    = flat_text[target_t]
    target_disp = _showable(tgt_char)

    def plot_heatmap(heat, title_suffix, png_name):
        masked = np.ma.masked_less_equal(heat, 0.0)
        thresholds = [lvl[0] for lvl in COLOR_LEVELS]
        colors     = [lvl[1] for lvl in COLOR_LEVELS]
        cmap = ListedColormap(colors)
        cmap.set_bad(color="#000000")
        norm = BoundaryNorm(thresholds, cmap.N, clip=True)

        fig, ax = plt.subplots(figsize=(W/4.0 if W>0 else 10, H/2.8 if H>0 else 5))
        im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        cbar = plt.colorbar(im)
        cbar.set_label("IG attribution (L2 norm; custom levels)")
        ticks = cbar.get_ticks()
        if len(ticks) > 0:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([">0"] + [f"{t:.3f}" for t in ticks[1:]])

        ax.set_title(
            f"{title_suffix}\n"
            f"target=({target_y},{target_x}) token: '{target_disp}'"
        )
        ax.set_xlabel(f"x (columns, 0 ... {max(0, W-1)})")
        ax.set_ylabel(f"y (rows, 0 ... {max(0, H-1)})")

        entries = []
        for y in range(H):
            row = level_lines[y]
            for x in range(min(W, len(row))):
                v = float(heat[y, x])
                if ANNOTATE_MODE in ("all", "topk", "threshold") and row:
                    entries.append((y, x, _showable(row[x]), v))

        cutoff = -1e9 if ANNOTATE_MODE == "all" else 0.0
        
        def text_color(v):
            high_thr = COLOR_LEVELS[-2][0]
            return "black" if v >= high_thr else "white"

        fs = max(6, int(13 - W / 85))

        for (y, x, ch, v) in entries:
            if v >= cutoff:
                ax.text(
                    x, y, ch, ha="center", va="center", fontsize=fs,
                    color=text_color(v),
                    path_effects=[pe.withStroke(linewidth=2, foreground="black", alpha=0.45)]
                )

        if 0 <= target_y < H and 0 <= target_x < W:
            rect = mpl.patches.Rectangle((target_x-0.5, target_y-0.5), 1, 1, fill=False, edgecolor="red", linewidth=2)
            ax.add_patch(rect)
            ax.text(
                target_x, target_y, _showable(level_lines[target_y][target_x]),
                ha="center", va="center", fontsize=max(fs, 12), color="red",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            )

        plt.tight_layout()
        fig.savefig(png_name, dpi=160)
        plt.close(fig)
        print(f"[SUCCESS] Heatmap saved -> {png_name}")

    png_embed = f"IG_{kind}_{target_x}-{target_y}_embed.png"
    plot_heatmap(heat_embed, "Embedding IG Analysis", png_embed)

    # ===== CSV =====
    import csv as _csv
    csv_embed = f"IG_{kind}_{target_x}-{target_y}_embed.csv"
    with open(csv_embed, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["y", "x", "char", "value"])
        for y in range(H):
            row_str = level_lines[y]
            row_len = len(row_str)
            for x in range(W):
                ch = row_str[x] if x < row_len else ""
                w.writerow([y, x, ch, float(heat_embed[y, x])])
    print(f"[SUCCESS] CSV saved -> {csv_embed}")

    return heat_embed, png_embed, csv_embed

# ========= Main =========
if __name__ == "__main__":
    set_seed(42)

    # 1) Initialize Model
    print(f"\nInitializing MoE MarioGPT (Experts={NUM_EXPERTS}, TopK={MOE_TOP_K})...")
    mario_lm = MarioLM(
        lm_path=BASE_MODEL,
        tokenizer_path=TOKENIZER_PATH,
        use_moe=True,
        num_experts=NUM_EXPERTS,
        moe_top_k=MOE_TOP_K,
    )

    weights_path = os.path.join(CHECKPOINT_DIR, "pytorch_model.bin")
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(weights_path, map_location=device)

        model_state = mario_lm.lm.state_dict()
        filtered, skipped = {}, []
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape != v.shape:
                skipped.append(f"  {k}: ckpt {tuple(v.shape)} -> model {tuple(model_state[k].shape)}")
            else:
                filtered[k] = v

        mario_lm.lm.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[WARN] Skipped {len(skipped)} shape-mismatched param(s) (randomly initialized):")
            for s in skipped:
                print(s)
        print("Weights loaded (partial).")
    else:
        print(f"Error: Checkpoint not found at {weights_path}")
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        mario_lm = mario_lm.to(device)
    
    hf_model = mario_lm.lm.eval()
    try:
        hf_model.config.use_cache = False
    except Exception:
        pass

    if ENABLE_GC and hasattr(hf_model, "gradient_checkpointing_enable"):
        try:
            hf_model.gradient_checkpointing_enable()
        except Exception:
            pass

    # 2) 三種類型的關卡字串對應表
    LEVEL_MAP = {
        "rect": generated_rect,
        "plat": generated_plat,
        "pipe": generated_pipe,
    }

    # 3) 依序對每種類型執行分析
    for kind, tgt in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"[種類: {kind}]  target=({tgt['y']}, {tgt['x']})")
        print(f"{'='*60}")

        level_text = LEVEL_MAP[kind].strip()
        if not level_text:
            print(f"[WARN] {kind} 關卡字串為空，跳過。")
            continue

        level_lines = level_text.splitlines()
        out_prefix  = f"ig_{kind}_X{tgt['x']}_Y{tgt['y']}"

        single_tile_ig_heatmap_sliding(
            mario_lm,
            level_lines=level_lines,
            target_y=tgt["y"],
            target_x=tgt["x"],
            kind=kind,
            out_prefix=out_prefix,
        )

    print("\n🎉 全部 IG 貢獻度分析已完成！")