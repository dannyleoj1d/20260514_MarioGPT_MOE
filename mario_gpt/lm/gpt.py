from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from mario_gpt.lm.base import BaseMarioLM
from mario_gpt.prompter import Prompter
from mario_gpt.sampler import GPTSampler, SampleOutput
from mario_gpt.lm.gpt2_2dpos_model import GPT2With2DSinusoids

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"

# --- MoE Components Definition Start ---

class MarioExpert(nn.Module):
    """
    Standard MLP acting as a single Expert.
    Mimics the structure of GPT2MLP but uses standard Linear layers.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MoEMLP(nn.Module):
    """
    Mixture of Experts 層
    Router 輸入 = [hidden_states(D)]
    """
    def __init__(self, config, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = config.n_embd
        self.intermediate_size = config.n_inner if config.n_inner is not None else 4 * self.hidden_size

        # Router 輸入維度：hidden only
        self.router = nn.Linear(self.hidden_size, num_experts, bias=False)

        self.last_expert_indices = None
        self.experts = nn.ModuleList([
            MarioExpert(self.hidden_size, self.intermediate_size, config.resid_pdrop)
            for _ in range(num_experts)
        ])

        self.aux_loss = 0.0

    def forward(self, hidden_states):
        # hidden_states: [B, T, D]
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        router_input_flat = hidden_states.view(-1, hidden_dim)  # [B*T, D]
        hidden_states_flat = router_input_flat

        # 1. Router Logic
        logits = self.router(router_input_flat)
        probs = F.softmax(logits, dim=-1)
        
        # 選擇 Top-K 專家
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        self.last_expert_indices = top_k_indices.detach().cpu()
        
        # Normalize
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 2. Auxiliary Loss (負載平衡)
        importance = probs.sum(0)
        target = router_input_flat.shape[0] / self.num_experts
        self.aux_loss = torch.mean((importance - target)**2) * 0.01

        # 3. Process through Experts
        final_output = torch.zeros_like(hidden_states_flat)
        
        for i, expert in enumerate(self.experts):
            # 檢查哪些 token 選擇了第 i 個專家
            mask = (top_k_indices == i).any(dim=-1)
            
            if mask.any():
                selected_tokens = hidden_states_flat[mask]
                expert_out = expert(selected_tokens)
                
                # 取得該專家對應的權重
                # top_k_indices[mask] 可能在 dim 1 的不同位置選中 i，需要精確提取
                locs = (top_k_indices[mask] == i).nonzero(as_tuple=True)[1]
                weights = top_k_probs[mask, locs].unsqueeze(-1)
                
                final_output[mask] += expert_out * weights

        return final_output.view(batch_size, sequence_length, hidden_dim)
# --- MoE Components Definition End ---


class MarioGPT(BaseMarioLM):
    PRETRAINED_LM_PATH = PRETRAINED_MODEL_PATH
    PRETRAINED_TOKENIZER_PATH = PRETRAINED_MODEL_PATH

    BASE_LM_PATH = "distilgpt2"
    BASE_TOKENIZER_PATH = "distilgpt2"

    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_len: int = 700,
        prompter: Optional[Prompter] = None,
        lm_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        lm_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        # New MoE Arguments
        use_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2,
    ):
        super().__init__(
            lm,
            tokenizer,
            context_len,
            lm_path,
            tokenizer_path,
            lm_kwargs,
            tokenizer_kwargs,
        )
        self.prompter = prompter
        if prompter is None:
            self.prompter = Prompter(self.tokenizer)

        # Apply MoE conversion if requested
        self.use_moe = use_moe
        if self.use_moe:
            print(f" Converting MarioGPT to Mixture of Experts (Experts: {num_experts}, Top-K: {moe_top_k})...")
            self.convert_layers_to_moe(num_experts, moe_top_k)
        
        from mario_gpt.lm.gpt2_2dpos_model import GPT2With2DSinusoids
        print(">>> After load, model class =", type(self.lm))
        assert isinstance(self.lm, GPT2With2DSinusoids), \
        f"Expected GPT2With2DSinusoids, got {type(self.lm)}"

    def get_last_routing_info(self):
        """
        Returns a list of tensors containing the expert indices selected 
        in the last forward pass for each MoE layer.
        """
        routing_info = []
        
        # 遍歷所有 Transformer Block
        if hasattr(self.lm, "transformer"):
            blocks = self.lm.transformer.h
        elif hasattr(self.lm, "h"):
            blocks = self.lm.h
        else:
            return []

        for i, block in enumerate(blocks):
            # 檢查 MLP 是否為我們自定義的 MoEMLP
            if hasattr(block, "mlp") and hasattr(block.mlp, "last_expert_indices"):
                # 結構: {'layer': 層數, 'indices': 選擇的專家索引}
                routing_info.append({
                    'layer': i,
                    'indices': block.mlp.last_expert_indices
                })
        
        return routing_info

    def convert_layers_to_moe(self, num_experts, top_k):
        """
        Replaces the standard MLP layers in the GPT2 model with MoE layers.
        """
        # Access the underlying transformer blocks
        # For GPT2, usually: self.lm.transformer.h
        if hasattr(self.lm, "transformer"):
            blocks = self.lm.transformer.h
        elif hasattr(self.lm, "h"): # DistilGPT2 sometimes structure
            blocks = self.lm.h
        else:
            print("Warning: Could not find transformer blocks to convert to MoE.")
            return

        config = self.lm.config
        
        for i, block in enumerate(blocks):
            # Locate the MLP layer. In HuggingFace GPT2, it's usually block.mlp
            if hasattr(block, "mlp"):
                # Initialize new MoE Layer
                moe_layer = MoEMLP(config, num_experts=num_experts, top_k=top_k)
                # Replace the existing MLP
                block.mlp = moe_layer
                # Optional: You might want to copy weights from the original MLP to the experts
                # to initialize them (Expert Upcycling), but here we initialize fresh.
            else:
                print(f"Warning: Block {i} does not have an 'mlp' attribute.")

    def get_aux_loss(self):
        """
        Aggregates the auxiliary loss from all MoE layers.
        Call this during training to add to the main loss.
        """
        if not self.use_moe:
            return 0.0
        
        total_aux_loss = 0.0
        count = 0
        
        # Navigate to blocks again
        if hasattr(self.lm, "transformer"):
            blocks = self.lm.transformer.h
        else:
            blocks = self.lm.h # Fallback
            
        for block in blocks:
            if hasattr(block, "mlp") and isinstance(block.mlp, MoEMLP):
                total_aux_loss += block.mlp.aux_loss
                count += 1
        
        return total_aux_loss if count > 0 else 0.0

    def generate_seed(self, length: int, batch_size: Optional[int] = None):
        seed = self.tokenizer("X", return_tensors="pt").input_ids.squeeze()
        if batch_size is None:
            return seed.repeat(length)
        return seed.view(1, 1).repeat(batch_size, length)

    def load_pretrained_lm(self, path: str, lm_kwargs: Dict[str, Any]) -> GPT2Model:
        if path == "random":
            print("Initializing random GPT2With2DSinusoids weights...")
            config = AutoConfig.from_pretrained(self.BASE_LM_PATH, **{**lm_kwargs, "add_cross_attention": True})
            model = GPT2With2DSinusoids.from_config(config)
            return model
        
        print(f"------ Loading pretrained model from {path} into GPT2With2DSinusoids")
        config = AutoConfig.from_pretrained(path, **{**lm_kwargs, "add_cross_attention": True})
        model = GPT2With2DSinusoids.from_pretrained(path, config=config)
        return model

    def load_pretrained_tokenizer(
        self, path: str, tokenizer_kwargs: Dict[str, Any]
    ) -> GPT2Tokenizer:
        if path == "random":
            return AutoTokenizer.from_pretrained(
                self.BASE_TOKENIZER_PATH, **tokenizer_kwargs
            )
        return AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

    def sample(
        self,
        seed: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
        num_steps: int = 1,
        temperature: float = 2.0,
        encoder_hidden_states: torch.Tensor = None,
        use_tqdm: bool = False,
        return_tensor: bool = False,
    ) -> SampleOutput:
        # 1. 實例化 Sampler
        sampler = GPTSampler(
            self, 
            temperature=temperature, 
            context_len=self.context_len, 
            use_tqdm=use_tqdm
        )
        
        # 2. [修正點] 顯式呼叫 .sample 方法，而不是呼叫物件本身
        return sampler.sample(
            seed=seed,
            prompts=prompts,
            num_steps=num_steps,
            encoder_hidden_states=encoder_hidden_states,
            return_tensor=return_tensor,
        )