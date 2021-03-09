import argparse
import os
from modules.data.read import DataReader, add_data_reader_arguments
import sys
from tqdm import tqdm
import torch


def filter_results(nr):
    return [x[:x.find("<|endoftext|>")][:x.find("</s>")] for x in nr]


def generate(model, text, additional_len=20):
    min_len = min(len(model.tokenizer.encode(text)), 2048 - additional_len)
    return filter_results(model.generate(
        text=text,
        max_length=min_len + additional_len,
        num_beams=10,
        eos_token_id=model.tokenizer.eos_token_id,
        num_return_sequences=1,
    ))[0]


def get_model(args):
    sys.path.append(args.ru_gpts_dir)
    os.environ["USE_DEEPSPEED"] = "1"
    from src.xl_wrapper import RuGPT3XL
    gpt = RuGPT3XL.from_pretrained(
        args.tokenizer_name,
        weights_path=args.weights_path,
        deepspeed_config_path=args.deepspeed_config_path,
        seq_len=512,
        master_port=args.master_port
    )
    gpt.tokenizer.add_special_tokens({"bos_token": "<s>"})
    gpt.tokenizer.add_special_tokens({"eos_token": "</s>"})
    if len(args.start_sep):
        gpt.tokenizer.add_special_tokens({"start_sep": args.start_sep})
    return gpt


def add_prediction_arguments(parser):
    group = parser.add_argument_group('predict', 'Prediction arguments.')
    group.add_argument(
        '--ru_gpts_dir',
        type=str,
        default="../ru-gpts/",
        help='directory for ru-gpts repo for load pretrained XL model'
    )
    group.add_argument(
        '--deepspeed_config_path',
        type=str,
        default="../ru-gpts/src/deepspeed_config/gpt3_xl_sparse_2048.json",
        help='path to deepspeed config for sparsity'
    )
    group.add_argument(
        '--master_port',
        type=str,
        default="6000",
        help='master port for parallel prediction'
    )
    group.add_argument(
        '--weights_path',
        type=str,
        help='path to pretrained weights'
    )
    group.add_argument(
        '--save_preds_path',
        type=str,
        default="../test_pred/",
        help='path for store predictions'
    )
    group.add_argument(
        '--total_processes',
        type=int,
        default=None,
        help='count of total used gpus'
    )
    group.add_argument(
        '--num_proc',
        type=int,
        default=0,
        help='number of current process'
    )
    return parser


def predict(reader, model, path, total_processes, num_proc):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    for data_part in reader.lm_prefixes:
        store_dir = os.path.join(path, data_part)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        names = list(reader.lm_prefixes[data_part].keys())

        # split to shards
        shard_size = len(names) // total_processes
        shard_start = num_proc * shard_size
        shard_end = (num_proc + 1) * shard_size
        names = names[shard_start:shard_end]

        for name in tqdm(names, total=len(names), leave=False, desc="Predict"):
            with open(os.path.join(store_dir, f"{name}.norm"), 'w', encoding='utf-8') as file:
                for lm_prefix, ann in tqdm(
                        zip(reader.lm_prefixes[data_part][name][:5], reader.anns[data_part][name]),
                        total=len((reader.lm_prefixes[data_part][name])),
                        leave=False
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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Predict on test set")
    arg_parser = add_data_reader_arguments(arg_parser)
    args = arg_parser.parse_args()
    print(f"Run {args.num_proc} proc")
    if args.total_processes is None:
        print("Note total processes is None. Suppose, that running on all gpus...")
        args.total_processes = torch.cuda.device_count()
    args_dict = vars(args)
    args_dict["data_parts"] = args_dict["data_parts"].split(",")
    print("Load data for predict...")
    reader = DataReader(**args_dict)
    reader.prc(is_save=True)

    print("Load model...")
    model = get_model(args)

    print(f"Start prediction on {args.num_proc} proc...")
    predict(reader, model, args.save_preds_path, args.total_processes, args.num_proc)
