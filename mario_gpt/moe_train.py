import torch
# 注意：如果你修改的是 gpt.py 中的 MarioGPT 類別，
# 且 mario_gpt/__init__.py 沒有將其導出為 MarioLM，
# 你可能需要改用: from mario_gpt.lm.gpt import MarioGPT as MarioLM
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer

TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"

# 1. 設定基礎模型 (通常是用 distilgpt2 或原本訓練好的 MarioGPT)
BASE = "random" 
# 或者用原本的訓練權重: "shyamsn97/Mario-GPT2-700-context-length"

# 2. 實例化模型並啟用 MoE
# 這裡是關鍵修改：加入 use_moe, num_experts, moe_top_k 參數
mario_lm = MarioLM(
    lm_path=BASE, 
    tokenizer_path=TOKENIZER_PATH,
    use_moe=True,       # <--- 啟用 Mixture of Experts
    num_experts=8,      # <--- 設定專家數量 (例如 8 個)
    moe_top_k=2         # <--- 每個 Token 選擇的前 K 個專家 (例如 2 個)
)

# 3. 建立資料集 (不變)
dataset = MarioDataset(mario_lm.tokenizer)

# 4. 建立訓練設定
# 建議顯式設定 aux_loss_coeff (MoE 負載平衡 Loss 的權重)，預設是 0.01
config = TrainingConfig(
    mixed_precision="bf16",
    save_iteration=10000,
    aux_loss_coeff=0.01, # <--- 確保專家負載平衡
    batch_size=4         # 根據你的顯存大小調整
)

# 5. 建立 Trainer
trainer = MarioGPTTrainer(mario_lm, dataset, config=config)

# 6. 開始訓練！
print(f"Start training MoE MarioGPT with {mario_lm.num_experts} experts...")
trainer.train(100000) # 訓練 100 個迭代