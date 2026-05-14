import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GPT2Config, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


def rotate_half(x):
    # x[..., 0::2], x[..., 1::2] 交錯為 (x1, x2)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # [-x2, x1]
    x_rot = torch.stack((-x2, x1), dim=-1)  # [..., D/2, 2]
    return x_rot.flatten(-2)                # [..., D]

class RopeGPT2Attention(GPT2Attention):
    """
    在 HuggingFace GPT2Attention 基礎上插入 RoPE 2D。
    用法：把 GPT2Model.transformer.h[i].attn 替換成本類別的實例。
    在 forward 前把 position_2d 掛到 self._rope_position_2d 供使用。
    """
    def __init__(self, config):
        super().__init__(config)

        embed_dim = getattr(self, 'split_size', None) or getattr(self, 'embed_dim', config.n_embd)
        self.head_dim = embed_dim // self.num_heads
        assert self.head_dim % 2 == 0, "head_dim 必須是偶數"

        # 分別為 X/Y 設不同頻率底數（可自行調）
        inv_freq_x = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        inv_freq_y = 1.0 / ( 5000.0 ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        self.register_buffer("inv_freq_x", inv_freq_x)  # [Dh/2]
        self.register_buffer("inv_freq_y", inv_freq_y)  # [Dh/2]

        # 供外部在每次 forward 前注入：shape [B, T, 2] (x, y)
        self._rope_position_2d = None

    def _apply_rope_2d(self, q, k, position_2d):
        """
        q, k: [B, nH, T, Dh]
        position_2d: [B, T, 2]，整個 batch 共用；或你也可以做每樣本不同
        """
        device = q.device
        B, nH, T, Dh = q.shape
        assert Dh == self.head_dim

        # 取 x,y（long → float），並擴成 [B, 1, T, Dh/2] 的角頻率
        x_pos = position_2d[..., 0].to(device=device, dtype=self.inv_freq_x.dtype)  # [B, T]
        y_pos = position_2d[..., 1].to(device=device, dtype=self.inv_freq_y.dtype)  # [B, T]

        # [T, Dh/2] ← [T,1] * [1,Dh/2]（這裡用 outer/einsum）
        freqs_x = torch.einsum("bt,d->btd", x_pos, self.inv_freq_x)  # [B,T,Dh/2]
        freqs_y = torch.einsum("bt,d->btd", y_pos, self.inv_freq_y)  # [B,T,Dh/2]

        # 擴成 [B,1,T,Dh/2] 以便 broadcast 到 nH
        cos_x = torch.cos(freqs_x).unsqueeze(1)  # [B,1,T,Dh/2]
        sin_x = torch.sin(freqs_x).unsqueeze(1)  # [B,1,T,Dh/2]
        cos_y = torch.cos(freqs_y).unsqueeze(1)  # [B,1,T,Dh/2]
        sin_y = torch.sin(freqs_y).unsqueeze(1)  # [B,1,T,Dh/2]

        # 把 head_dim 二等分：前半專給 X、後半專給 Y
        qx, qy = q[..., :Dh//2], q[..., Dh//2:]  # [B,nH,T,Dh/2]
        kx, ky = k[..., :Dh//2], k[..., Dh//2:]

        # 標準 RoPE：x*cos + rotate_half(x)*sin
        qx_rot = (qx * cos_x) + (rotate_half(qx) * sin_x)
        qy_rot = (qy * cos_y) + (rotate_half(qy) * sin_y)
        kx_rot = (kx * cos_x) + (rotate_half(kx) * sin_x)
        ky_rot = (ky * cos_y) + (rotate_half(ky) * sin_y)

        q_rot = torch.cat([qx_rot, qy_rot], dim=-1)
        k_rot = torch.cat([kx_rot, ky_rot], dim=-1)
        return q_rot, k_rot

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        return tensor.view(new_shape).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        position_2d=None,
        **kwargs,  # <--- 加上這個，接收包含 encoder_hidden_states 在內的所有多餘參數
    ):
        embed_dim = hidden_states.size(-1)
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(embed_dim, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key   = self._split_heads(key,   self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # === [修改 2] 強化版 RoPE-2D 注入邏輯 ===
        # 優先使用直接傳入的 position_2d，其次才使用掛載在 self 上的預存座標
        rope_pos = position_2d if position_2d is not None else self._rope_position_2d
        
        if rope_pos is not None:
            # [關鍵] 處理 KV Cache 模式下的座標對齊
            # 如果是生成模式，query 長度通常為 1，但 rope_pos 長度是全序列長度
            # 我們必須取 rope_pos 的最後一個位置給 query 旋轉
            q_len = query.size(2)
            k_len = key.size(2)
            
            # 針對 Query：如果是增量生成 (q_len=1)，取座標序列的最後一項
            # 針對 Key：在旋轉時應使用當前最新的座標片段
            q_rope_pos = rope_pos[:, -q_len:, :]
            
            # 執行旋轉 (注意：RoPE 應在拼接 past_key 之前對當前 key 進行旋轉)
            query, key = self._apply_rope_2d(query, key, q_rope_pos)

        # past kv cache (注意：在 RoPE 之後拼接，因為 cache 裡的 key 應該是已經轉過的)
        if layer_past is not None:
            past_key, past_value = layer_past
            key   = torch.cat((past_key,   key),   dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        # 手動注意力計算（相容 transformers 5.x）
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        # 套用 Causal Mask（self.bias 是 GPT2Attention 繼承來的三角遮罩 buffer）
        q_len, k_len = query.size(-2), key.size(-2)
        if hasattr(self, 'bias'):
            causal_mask = self.bias[:, :, k_len - q_len : k_len, :k_len]
            mask_value = torch.finfo(attn_weights.dtype).min
            attn_weights = torch.where(causal_mask.bool(), attn_weights,
                                       torch.full_like(attn_weights, mask_value))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        if hasattr(self, 'attn_dropout'):
            attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        if hasattr(self, 'resid_dropout'):
            attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

#GPT2WithRoPE2D
#GPT2With2DSinusoids
class GPT2With2DSinusoids(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 1. 預先替換 Attention 為 RoPE 版本
        for i, block in enumerate(self.transformer.h):
            rope_attn = RopeGPT2Attention(config)
            rope_attn.load_state_dict(block.attn.state_dict(), strict=False)
            self.transformer.h[i].attn = rope_attn

        self.post_init()

    # 必須保留：與 Transformers 庫對接的接口
    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_2d=None,              # [B, T, 2] 核心 2D 座標
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,    # [B, N, 768] from BART prompt encoder
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # 初始化參數
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. 處理輸入 Embedding
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.transformer.wte(input_ids)
        
        # 2. 獲取 GPT2 內部的因果遮罩 (Causal Mask)
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        
        extended_attention_mask = self.transformer.get_extended_attention_mask(
            attention_mask, input_shape, dtype=inputs_embeds.dtype
        )

        # 3. 手動分發座標到每一層
        hidden_states = inputs_embeds
        presents = [] if use_cache else None
        
        for block in self.transformer.h:
            # --- Attention 階段 ---
            residual = hidden_states
            hidden_states = block.ln_1(hidden_states)
            block.attn._rope_position_2d = position_2d
            
            attn_outputs = block.attn(
                hidden_states,
                attention_mask=extended_attention_mask,
                use_cache=use_cache,
                **kwargs
            )
            attn_output = attn_outputs[0]
            if use_cache:
                presents.append(attn_outputs[1])
            
            hidden_states = residual + attn_output

            # --- Cross-Attention 階段 ---
            # block.crossattention 存在於 add_cross_attention=True 的 GPT2Block 中
            # 此處呼叫確保梯度在 loss.backward() 時能流入 crossattention 權重
            if encoder_hidden_states is not None and hasattr(block, 'crossattention'):
                residual = hidden_states
                hidden_states = block.ln_cross_attn(hidden_states)
                cross_attn_output = block.crossattention(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )[0]
                hidden_states = residual + cross_attn_output

            # --- MoE MLP 階段 ---
            residual = hidden_states
            hidden_states = block.ln_2(hidden_states)

            feed_forward_hidden_states = block.mlp(hidden_states)
            
            hidden_states = residual + feed_forward_hidden_states

        # 4. 輸出層
        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        # 5. Loss 計算
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )


        if not return_dict:
            return (logits, presents)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=None,
            attentions=None,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(config, **kwargs)