import argparse
import os

from tqdm import tqdm
from transformers import GPT2Tokenizer

from modules.utils import get_all_files_from_dir
from collections import defaultdict


class DataReader(object):
    def __init__(
            self,
            path,
            tokenizer_name,
            output_dir,
            part="train",
            train_part_name="train",
            data_parts=frozenset(["generic", "named"]),
            answer_sep="<answer>",
            start_sep="<start>",
            end_sep="<end>",
            window_size=0,
            local_rank=0,
            word_size=1,
            **kwargs
    ):
        self.part = part
        self.path = path
        self.train_part_name = train_part_name
        self.data_parts = data_parts
        self.texts = defaultdict(dict)
        self.anns = defaultdict(dict)
        self.norms = defaultdict(dict)
        self.lm_prefixes = defaultdict(dict)
        self.tokenizer_name = tokenizer_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens({"bos_token": "<s>"})
        self.tokenizer.add_special_tokens({"eos_token": "</s>"})

        print("Add answer_sep:", answer_sep)
        self.tokenizer.add_tokens(answer_sep)
        self.answer_sep = answer_sep

        print("Add start_sep", start_sep)
        self.start_sep = start_sep
        self.tokenizer.add_tokens(start_sep)

        print("Add start_sep", end_sep)
        self.tokenizer.add_tokens(end_sep)
        self.end_sep = end_sep

        self.window_size = window_size

        self.output_dir = output_dir
        self.file_list_name = "files.list"
        self.files_dir_name = "files"
        os.makedirs(os.path.join(self.output_dir, self.files_dir_name), exist_ok=True)
        self.texts_for_model = []

        self.local_rank = local_rank
        self.word_size = word_size

    def __call__(self):
        self.prc()

    def prc(self, is_save=True):
        self._read_files()
        self._make_raw_data(is_save=is_save)

    def _read_files(self):
        for data_part in self.data_parts:
            all_files = [x for x in get_all_files_from_dir(os.path.join(self.path, data_part)) if x.endswith(".txt")]
            shard_size = len(all_files) // self.word_size
            shard_start = self.local_rank * shard_size
            shard_end = (self.local_rank + 1) * shard_size
            all_files = all_files[shard_start:shard_end]
            for fn in tqdm(
                    all_files, total=len(all_files), leave=True,
                    desc=f"Parsing files from {self.part} part on {self.local_rank}..."
            ):
                name = os.path.basename(fn)
                name, ext = os.path.splitext(name)

                with open(fn, encoding='utf-8') as file_obj:
                    text = file_obj.read()
                self.texts[data_part][name] = text

                with open(fn[:-4] + ".ann", encoding='utf-8') as file_obj:
                    ann = file_obj.read().strip().split('\n')
                self.anns[data_part][name] = ann
                if self.part == self.train_part_name:
                    path = fn
                    p1, p2 = os.path.split(path)

                    path = os.path.join(p1.replace("texts_and_ann", ""), "norm", p2)
                    with open(path[:-4] + ".norm", encoding='utf-8') as file_obj:
                        norm = file_obj.read().strip().split('\n')
                    self.norms[data_part][name] = norm

    def _make_raw_data(self, is_save=True):
        file_list = None
        if is_save:
            file_list = open(os.path.join(self.output_dir, self.file_list_name), "w")
        for data_part in self.data_parts:
            for name in tqdm(
                    self.texts[data_part], total=len(self.texts[data_part]),
                    leave=True, desc="Making raw files..."):
                text = self.texts[data_part][name]
                if self.answer_sep in text:
                    print("answer_sep in file", name)
                ann = self.anns[data_part][name]
                if self.part == self.train_part_name:
                    norm = self.norms[data_part][name]
                    if len(norm) != len(ann):
                        raise ValueError(f"Lens of norm and ann should be same. Error with file name {name}")
                else:
                    norm = ["0"] * len(ann)
                file_texts = []
                for line_ann, line_norm in zip(ann, norm):
                    start, stop = list(map(int, line_ann.strip().split()))
                    line_norm = line_norm.strip()
                    if self.part == self.train_part_name:
                        final_text = "{bos}{text}{answ_sep}{ln}{eos}"
                    else:
                        final_text = "{bos}{text}{answ_sep}"
                    if self.window_size:
                        # left = self.tokenizer.decode(self.tokenizer.encode(text[:start])[-self.window_size:]).strip()
                        left = " ".join(text[:start].split()[-self.window_size:]).strip()
                        # right = self.tokenizer.decode(self.tokenizer.encode(text[stop:])[:self.window_size]).strip()
                        right = " ".join(text[stop:].split()[:self.window_size])
                        new_text = "{left}{start}{to_norm}{end}{right}".format(
                            left=left,
                            start=self.start_sep,
                            to_norm=text[start:stop],
                            end=self.end_sep,
                            right=right
                        )
                    else:
                        new_text = text[:start] + self.start_sep + text[start:stop]
                    new_text = new_text.replace("\u2028", " ").strip()
                    final_text = final_text.format(
                        bos=self.tokenizer.bos_token,
                        text=new_text,
                        answ_sep=self.answer_sep,
                        ln=line_norm,
                        eos=self.tokenizer.eos_token,
                    )
                    file_texts.append(final_text)
                self.lm_prefixes[data_part][name] = file_texts

                res_fn = os.path.join(self.output_dir, self.files_dir_name, f"{name}.txt")
                res_fn = os.path.abspath(res_fn)
                final_file_text = "\n".join(file_texts)
                if is_save:
                    file_list.write(f"{res_fn}\n")
                    if self.tokenizer_name == "sberbank-ai/rugpt3xl":
                        final_file_text += "<|endoftext|>"
                    with open(res_fn, "w", encoding='utf-8') as file_out:
                        file_out.write(final_file_text)
                self.texts_for_model.append(final_file_text)
        if is_save:
            file_list.close()


def add_data_reader_arguments(parser):
    group = parser.add_argument_group('data_reader', 'Data reader paths for processing raw files.')
    group.add_argument(
        '--path',
        type=str,
        help='path to the data dir'
    )
    group.add_argument(
        '--tokenizer_name',
        type=str,
        default="sberbank-ai/rugpt3xl",
        help='path or name of tokenizer for loading from transformers'
    )
    group.add_argument(
        '--output_dir',
        type=str,
        help='path to the data dir'
    )
    group.add_argument(
        '--part',
        type=str,
        default="train",
        help='partition name of data: train or test'
    )
    group.add_argument(
        '--train_part_name',
        type=str,
        default="train",
        help='name of train data partition.'
    )
    group.add_argument(
        '--data_parts',
        type=str,
        default="generic,named",
        help='names of directories in data: generic,named.'
    )
    group.add_argument(
        '--answer_sep',
        type=str,
        default="<answer>",
        help='separator between query and answer.'
    )
    group.add_argument(
        '--window_size',
        type=int,
        default=0,
        help='Use only left context or left and right. If 0 use full left context.'
    )
    group.add_argument(
        '--start_sep',
        type=str,
        default="<start>",
        help='separator for mark start of norm.'
    )
    group.add_argument(
        '--end_sep',
        type=str,
        default="<end>",
        help='separator for mark end of norm.'
    )
    group.add_argument(
        '--save_preds_path',
        type=str,
        default="../test_pred/",
        help='path for store predictions'
    )
    group.add_argument(
        '--do_sample',
        type=int,
        default=None,
        help='do sample while generation'
    )
    group.add_argument(
        '--num_beams',
        type=int,
        default=None,
        help='number of beams while generation'
    )
    group.add_argument('--loss_only_norm', action='store_true',
                       help='Calculate loss only on norms')
    group.add_argument('--line_by_line', action='store_true',
                       help='Is learn line by line with pad.')
    return parser


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Prepare data for language modeling")
    arg_parser = add_data_reader_arguments(arg_parser)
    args = arg_parser.parse_args()
    args = vars(args)
    args["data_parts"] = args["data_parts"].split(",")
    reader = DataReader(**args)
    reader()
