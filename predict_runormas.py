# coding=utf-8
# Copyright (c) 2020, Sber.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT3"""

import os

import torch
import torch.distributed as dist
from apex.optimizers import FusedAdam as Adam
from tqdm import tqdm

from modules.data.read import DataReader
from src import mpu
from src.arguments import get_args
from src.fp16 import FP16_Module
from src.fp16 import FP16_Optimizer
from src.learning_rates import AnnealingLR
from src.model import GPT3Model
from src.model import gpt3_get_params_for_weight_decay_optimization
from src.utils import (
    Timers, load_checkpoint, load_huggingface_model,
    print_args, print_rank_0,
    get_sparse_attention_config, DEEPSPEED_WRAP
)
from src.xl_wrapper import RuGPT3XL
from .pretrain_gpt3 import initialize_distributed, set_random_seed

# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = False
if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from src.model import DistributedDataParallel as DDP


def get_model(args):
    """Build the model."""

    print_rank_0('building GPT3 model ...')
    assert args.num_attention_heads % args.model_parallel_size == 0
    num_local_heads = args.num_attention_heads // args.model_parallel_size
    deepspeed_sparsity_config = None
    if DEEPSPEED_WRAP and args.deepspeed:
        deepspeed_sparsity_config = get_sparse_attention_config(args, num_local_heads)
    if deepspeed_sparsity_config is not None:
        print_rank_0(f"Use sparse attention with mode {args.sparse_mode}")
    model = GPT3Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=True,
                      deepspeed_sparsity_config=deepspeed_sparsity_config,
                      sparse_mode=args.sparse_mode)

    if args.load_huggingface is not None:
        model = load_huggingface_model(model, args.load_huggingface, args.huggingface_double_pos_embeddings)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if DEEPSPEED_WRAP and args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    param_groups = gpt3_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if DEEPSPEED_WRAP and args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               min_lr=args.min_lr)

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if DEEPSPEED_WRAP and args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = DEEPSPEED_WRAP.deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False
        )

    if args.load is not None:
        print_rank_0("Load checkpoint from " + args.load)
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args,
                                         deepspeed=DEEPSPEED_WRAP and args.deepspeed)
        print_rank_0("Checkpoint loaded")
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def get_reader(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""
    reader = None

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        world_size = mpu.get_data_parallel_world_size()
        rank = mpu.get_data_parallel_rank()
        # global_batch_size = args.batch_size * world_size

        args.data_parts = args.data_parts.split(",")
        reader = DataReader(**vars(args), local_rank=rank, word_size=world_size)
        reader.prc(is_save=False)

        tokenizer_path = args.tokenizer_path
        print_rank_0('Load tokenizer from ' + tokenizer_path)
        tokenizer = reader.tokenizer
        eod_token = tokenizer.encoder['<pad>']
        num_tokens = len(tokenizer)
        before = num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0(
            '> padded vocab (size: {}) with {} dummy tokens (new size: {})'.format(before, after - before, after))
        print_rank_0('> end-of-document token: {}'.format(eod_token))
        token_counts = torch.cuda.LongTensor(
            [after, eod_token, int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        tokenizer = None
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    eod_token = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    return reader, num_tokens, eod_token, tokenizer


def filter_results(nr):
    return [x[:x.find("<|endoftext|>")][:x.find("</s>")] for x in nr]


def generate(model, text, additional_len=32):
    min_len = min(len(model.tokenizer.encode(text)), 2048 - additional_len)
    return filter_results(model.generate(
        text=text,
        max_length=min_len + additional_len,
        num_beams=10,
        eos_token_id=model.tokenizer.eos_token_id,
        num_return_sequences=1,
    ))[0]


def predict(reader, model, path, num_proc):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    for data_part in reader.lm_prefixes:
        store_dir = os.path.join(path, data_part)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        names = list(reader.lm_prefixes[data_part].keys())

        for name in tqdm(names, total=len(names), leave=True, desc=f"Predict on {num_proc}"):
            with open(os.path.join(store_dir, f"{name}.norm"), 'w', encoding='utf-8') as file:
                for lm_prefix, ann in tqdm(
                        zip(reader.lm_prefixes[data_part][name], reader.anns[data_part][name]),
                        total=len((reader.lm_prefixes[data_part][name])),
                        leave=True
                ):
                    gen_res = generate(model, lm_prefix)
                    gen_res = gen_res.split(reader.answer_sep)
                    if len(gen_res) == 1:
                        text = reader.texts[data_part][name]
                        start, stop = list(map(int, ann.split()))
                        gen_res = text[start:stop].strip()
                    else:
                        gen_res = gen_res[1].strip()
                    file.write(f"{gen_res}\n")


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Predict RuNormAS')
        print_args(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    reader, args.vocab_size, args.eod_token, tokenizer = get_reader(args)

    # Model, optimizer, and learning rate.
    model_in, _, _ = setup_model_and_optimizer(args)

    model = RuGPT3XL(model_in, tokenizer, args.load, seq_len=512, min_generated_len=32)

    predict(reader, model, args.save_preds_path, mpu.get_data_parallel_rank())


if __name__ == "__main__":
    main()
