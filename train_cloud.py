import argparse
import torch
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train MoE MarioGPT on cloud GPU")
    parser.add_argument("--output_dir",   type=str,   default="./output",  help="Checkpoint save directory")
    parser.add_argument("--batch_size",   type=int,   default=8,            help="Training batch size (A100: 16~32)")
    parser.add_argument("--total_steps",  type=int,   default=100000,       help="Total training steps")
    parser.add_argument("--save_iter",    type=int,   default=10000,        help="Save checkpoint every N steps")
    parser.add_argument("--num_experts",  type=int,   default=8,            help="Number of MoE experts")
    parser.add_argument("--top_k",        type=int,   default=2,            help="Top-K experts per token")
    parser.add_argument("--aux_coeff",    type=float, default=0.01,         help="MoE load-balancing loss weight")
    parser.add_argument("--lr",           type=float, default=5e-4,         help="Learning rate")
    parser.add_argument("--mixed_precision", type=str, default="bf16",      help="bf16 / fp16 / no")
    parser.add_argument("--base_model",   type=str,   default="random",     help="'random' or HF model path")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 50)
    print(f"  MoE MarioGPT Cloud Training")
    print(f"  GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  batch_size   = {args.batch_size}")
    print(f"  total_steps  = {args.total_steps}")
    print(f"  num_experts  = {args.num_experts}  top_k = {args.top_k}")
    print(f"  output_dir   = {args.output_dir}")
    print("=" * 50)

    TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"

    mario_lm = MarioLM(
        lm_path=args.base_model,
        tokenizer_path=TOKENIZER_PATH,
        use_moe=True,
        num_experts=args.num_experts,
        moe_top_k=args.top_k,
    )

    dataset = MarioDataset(mario_lm.tokenizer)

    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        save_iteration=args.save_iter,
        aux_loss_coeff=args.aux_coeff,
        learning_rate=args.lr,
        mixed_precision=args.mixed_precision,
    )

    trainer = MarioGPTTrainer(mario_lm, dataset, config=config)
    trainer.train(args.total_steps)

if __name__ == "__main__":
    main()
