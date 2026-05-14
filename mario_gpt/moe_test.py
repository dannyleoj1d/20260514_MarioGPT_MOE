import torch
import os
import collections # 新增：用來統計次數
from mario_gpt import MarioLM, SampleOutput

# ================= 設定區 =================
# 1. 設定 MoE 參數 (必須與訓練時完全一致！)
NUM_EXPERTS = 8
MOE_TOP_K = 2

TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"

# 2. 設定模型路徑
CHECKPOINT_DIR = "Mario-GPT2-700-context-length/iteration_99999" 
BASE_MODEL = "random" 

# ================= 初始化模型 =================

print(f"Initializing MoE MarioGPT (Experts={NUM_EXPERTS}, TopK={MOE_TOP_K})...")

mario_lm = MarioLM(
    lm_path=BASE_MODEL,       
    tokenizer_path=TOKENIZER_PATH,
    use_moe=True,             
    num_experts=NUM_EXPERTS,  
    moe_top_k=MOE_TOP_K       
)

# 步驟 B: 載入訓練好的權重
weights_path = os.path.join(CHECKPOINT_DIR, "pytorch_model.bin")
if os.path.exists(weights_path):
    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=mario_lm.device)
    
    try:
        mario_lm.lm.load_state_dict(state_dict, strict=True)
        print("Success! Trained MoE weights loaded.")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        print("提示：請檢查 NUM_EXPERTS 是否與訓練時設定的數字一致。")
        exit()
else:
    print(f"Error: Checkpoint not found at {weights_path}")
    exit()

# 使用 CUDA 加速
if torch.cuda.is_available():
    device = torch.device('cuda')
    mario_lm = mario_lm.to(device)
    print("Moved model to CUDA.")

mario_lm.eval()

# ================= 開始生成 =================

prompts = ["no pipes, some enemies, some blocks, some rects, little coins, high elevation"]
print(f"Generating level with prompt: {prompts[0]}")

# 生成關卡
generated_level = mario_lm.sample(
    prompts=prompts,
    num_steps=1400,
    temperature=2.0, 
    use_tqdm=True
)

# 儲存結果
output_img = "moe_generated_level.png"
generated_level.img.save(output_img)
print(f"Image saved to {output_img}")

output_txt = "moe_generated_level.txt"
generated_level.save(output_txt)
print(f"Text level saved to {output_txt}")


# ================= 🔥 新增：專家選擇分析區 🔥 =================
print("\n" + "="*30)
print("   MOE ROUTING ANALYSIS   ")
print("="*30)

# 1. 取得歷史紀錄
history = generated_level.expert_choices

if history is None or len(history) == 0:
    print("警告：沒有找到專家選擇紀錄。請確認 sampler.py 和 gpt.py 是否已正確修改。")
else:
    print(f"成功取得 {len(history)} 步的生成紀錄。\n")

    # --- 分析 1: 查看第一步的詳細選擇 (Debugging) ---
    print("--- Step 0 Routing Details ---")
    first_step = history[0] # List of layers info
    for layer_info in first_step:
        layer_idx = layer_info['layer']
        # indices 形狀通常是 [Batch, TopK] -> [1, 2]
        indices = layer_info['indices'].tolist() 
        print(f"Layer {layer_idx}: Selected Experts {indices}")
    
    # --- 分析 2: 統計每一層的專家使用率 (Statistics) ---
    print("\n--- Expert Usage Distribution per Layer ---")
    
    # 初始化統計器
    layer_stats = collections.defaultdict(collections.Counter)
    total_stats = collections.Counter()
    
    # 遍歷所有步驟 (1400 steps)
    for step in history:
        for layer_info in step:
            layer_idx = layer_info['layer']
            # 把 Tensor 轉成 list 並展平: [[0, 5]] -> [0, 5]
            indices = layer_info['indices'].flatten().tolist()
            
            layer_stats[layer_idx].update(indices)
            total_stats.update(indices)
            
    # 印出每一層的統計結果
    # 假設模型有 6 層 (DistilGPT2 預設)
    sorted_layers = sorted(layer_stats.keys())
    for layer_idx in sorted_layers:
        counts = layer_stats[layer_idx]
        total_calls = sum(counts.values())
        print(f"\n[Layer {layer_idx}] Total calls: {total_calls}")
        
        # 依專家編號排序印出
        for expert_id in range(NUM_EXPERTS):
            count = counts[expert_id]
            percentage = (count / total_calls) * 100 if total_calls > 0 else 0
            bar = "#" * int(percentage / 5) # 簡易長條圖
            print(f"  Expert {expert_id}: {count:5d} ({percentage:5.1f}%) {bar}")

    # --- 分析 3: 整體專家偏好 ---
    print("\n--- Overall Expert Usage (All Layers) ---")
    grand_total = sum(total_stats.values())
    for expert_id in range(NUM_EXPERTS):
        count = total_stats[expert_id]
        percentage = (count / grand_total) * 100 if grand_total > 0 else 0
        print(f"Expert {expert_id}: {percentage:5.1f}%")

print("\nAnalysis Complete.")