from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Any

import numpy as np
import torch
from PIL.Image import Image
from tqdm import tqdm
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper

from mario_gpt.lm.base import BaseMarioLM
from mario_gpt.prompter import Prompter
from mario_gpt.simulator import Simulator
from mario_gpt.dataset import build_positions_2d  # [新增] 導入座標構建函數
from mario_gpt.utils import (
    convert_level_to_png,
    load_level,
    save_level,
    trim_level,
    view_level,
)

@dataclass
class SampleOutput:
    level: Optional[List[str]]
    prompt: Optional[str] = None
    img: Optional[Image] = None
    sample_predictions_str: Optional[List[str]] = None
    sample_predictions_img: Optional[Image] = None
    level_tensor: Optional[torch.Tensor] = None
    sample_predictions_tensor: Optional[torch.Tensor] = None
    expert_choices: Optional[List[Any]] = None

    @classmethod
    def create(
        cls,
        level_tensor: torch.Tensor,
        sample_predictions_tensor: torch.Tensor,
        tokenizer,
        prompter: Optional[Prompter] = None,
        expert_choices: Optional[List[Any]] = None,
    ) -> SampleOutput:
        level = None
        img = None
        try:
            level = view_level(level_tensor, tokenizer)
            img = convert_level_to_png(level)[0]
        except Exception as e:
            print(f"Failed representation! {e}")
        
        try:
            sample_predictions_str = view_level(sample_predictions_tensor, tokenizer)
            sample_predictions_img = convert_level_to_png(sample_predictions_str)[0]
        except Exception as e:
            sample_predictions_str = None
            sample_predictions_img = None

        prompt = None
        if prompter is not None:
            prompt = prompter(level_tensor)[0]

        return SampleOutput(
            level, prompt, img, sample_predictions_str, sample_predictions_img,
            level_tensor, sample_predictions_tensor, expert_choices,
        )
    
    def save(self, filename: str) -> str:
        from mario_gpt.utils import save_level
        save_level(self.level, filename)
        return filename

    @classmethod
    def from_level_predictions(
        cls,
        level: torch.Tensor,
        sample_predictions: torch.Tensor,
        tokenizer,
        prompter: Optional[Prompter] = None,
    ) -> Union[SampleOutput, List[SampleOutput]]:
        level_tensor = trim_level(level).squeeze().detach().cpu()
        sample_predictions_tensor = trim_level(sample_predictions).squeeze().detach().cpu()

        if len(level_tensor.shape) == 1:
            return SampleOutput.create(level_tensor, sample_predictions_tensor, tokenizer, prompter)

        out = []
        for _level_tensor, _sample_predictions_tensor in zip(level_tensor, sample_predictions_tensor):
            out.append(SampleOutput.create(_level_tensor, _sample_predictions_tensor, tokenizer, prompter))
        return out

class GPTSampler:
    def __init__(self, mario_lm: BaseMarioLM, temperature: float = 2.0, top_k: int = 16, context_len: int = 700, use_tqdm: bool = False, use_argmax: bool = False):
        self.mario_lm = mario_lm
        self.temperature = temperature
        self.top_k = top_k
        self.context_len = context_len
        self.use_tqdm = use_tqdm
        self.use_argmax = use_argmax
        self.logits_processor = LogitsProcessorList()
        self.logits_warper = LogitsProcessorList([TopKLogitsWarper(top_k), TemperatureLogitsWarper(temperature)])

    @property
    def device(self) -> torch.device:
        return self.mario_lm.device

    def step(
        self,
        seed: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            attention_mask = torch.ones_like(seed).to(seed.device)
            out = self.mario_lm.lm(
                input_ids=seed,
                attention_mask=attention_mask,
                position_2d=position_2d,
                encoder_hidden_states=encoder_hidden_states,
                token_type_ids=None,
            )
            logits = out.logits.detach()
            if len(logits.shape) == 2:
                logits = logits.view(1, 1, -1)
            next_token_logits = logits[:, -1, :]

            if self.use_argmax:
                next_tokens = next_token_logits.argmax(-1)
            else:
                next_token_scores = self.logits_processor(seed, next_token_logits)
                next_token_scores = self.logits_warper(seed, next_token_scores)
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens, encoder_hidden_states

    def sample(
        self,
        seed: Union[Optional[torch.Tensor], Optional[SampleOutput]] = None,
        prompts: Optional[List[str]] = None,
        num_steps: int = 1,
        encoder_hidden_states: torch.Tensor = None,
        return_tensor: bool = False,
    ):
        self.mario_lm.eval()
        context_len = self.context_len - 28
        generation_routing_history = [] 

        with torch.no_grad():
            if seed is None:
                seed = self.mario_lm.generate_seed(1, batch_size=len(prompts)).to(self.device)
                out_tensor = seed.to(self.device)
            elif isinstance(seed, SampleOutput):
                out_tensor = seed.level_tensor.to(self.device).squeeze()
            else:
                out_tensor = seed.to(self.device).squeeze()
            
            if len(out_tensor.shape) < 2:
                out_tensor = out_tensor.view(1, -1).repeat(len(prompts), 1)

            if encoder_hidden_states is None:
                if prompts is not None:
                    encoder_hidden_states = torch.stack([self.mario_lm.prompter.output_hidden(p) for p in prompts])
                else:
                    encoder_hidden_states = torch.stack([self.mario_lm.prompter(sample_prompt=True)[1] for _ in range(out_tensor.shape[0])])
            
            encoder_hidden_states = encoder_hidden_states.to(self.device).view(out_tensor.shape[0], 1, -1)
            
            bar = tqdm(np.arange(num_steps)) if self.use_tqdm else np.arange(num_steps)

            for i in bar:
                curr_len = out_tensor.shape[-1]
                
                # 1. 生成基於當前序列總長度的完整 2D 座標
                full_pos = build_positions_2d(curr_len, height=14, device=self.device)
                
                inp = out_tensor * 1
                pos_slice = full_pos.clone().float()

                # 2. [修改重點] 滑動視窗對齊邏輯與「相對座標化」
                if curr_len > context_len:
                    diff = curr_len % 14
                    ctx = context_len + diff
                    inp = inp[:, -ctx:]
                    
                    # 同步切片座標
                    pos_slice = full_pos[-ctx:, :].clone().float()

                    # === [核心修改：相對座標轉換] ===
                    # 獲取當前視窗中最左側點的 X 座標
                    min_x = pos_slice[0, 0].item()
                    # 將整個視窗的 X 座標減去最小值，使其始終從 0 開始
                    pos_slice[:, 0] -= min_x
                    # ============================
                
                # [可選] 座標歸一化：若您訓練時有除以寬度，請在此同步處理
                # pos_slice[:, 0] /= 700.0 
                # pos_slice[:, 1] /= 14.0

                # 準備傳入 step 的 Batch 座標 [B, T, 2]
                batch_pos = pos_slice.unsqueeze(0).repeat(out_tensor.shape[0], 1, 1)

                # 模型預測下一個 Token
                next_tokens, encoder_hidden_states = self.step(
                    inp,
                    encoder_hidden_states=encoder_hidden_states,
                    position_2d=batch_pos,
                )
                
                if hasattr(self.mario_lm, "get_last_routing_info"):
                    generation_routing_history.append(self.mario_lm.get_last_routing_info())

                out_tensor = torch.cat([out_tensor, next_tokens.unsqueeze(-1)], dim=-1)
                
                if self.use_tqdm:
                    bar.set_description(f"Len: {out_tensor.shape[-1]}")

        sample_out = SampleOutput.from_level_predictions(
            out_tensor, out_tensor[:, -num_steps:],
            self.mario_lm.tokenizer, self.mario_lm.prompter
        )

        if isinstance(sample_out, list):
            for out in sample_out: out.expert_choices = generation_routing_history
        else:
            sample_out.expert_choices = generation_routing_history

        self.mario_lm.train()
        return (sample_out, out_tensor) if return_tensor else sample_out