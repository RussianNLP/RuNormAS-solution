import argparse
import os

from tqdm import tqdm
from transformers import GPT2Tokenizer

from src.utils import get_all_files_from_dir


class DataReader(object):
    def __init__(
            self,
            path,
            tokenizer_name,
            output_dir,
            part="train",
            train_part_name="train",
            data_parts=frozenset(["generic", "named"]),
            answer_sep=" A: "
    ):
        self.part = part
        self.path = path
        self.train_part_name = train_part_name
        self.data_parts = data_parts
        self.texts = {}
        self.anns = {}
        self.norms = {}
        self.lm_prefixes = {}
        self.tokenizer_name = tokenizer_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens({"bos_token": "<s>"})
        self.tokenizer.add_special_tokens({"eos_token": "</s>"})
        self.answer_sep = answer_sep
        self.output_dir = output_dir
        self.file_list_name = "files.list"
        self.files_dir_name = "files"
        os.makedirs(os.path.join(self.output_dir, self.files_dir_name), exist_ok=True)
        self.texts_for_model = []

    def __call__(self):
        self.prc()

    def prc(self, is_save=True):
        self._read_files()
        self._make_raw_data(is_save=is_save)

    def _read_files(self):
        for data_part in self.data_parts:
            all_files = get_all_files_from_dir(os.path.join(self.path, data_part))
            for fn in tqdm(
                    all_files, total=len(all_files), leave=False,
                    desc=f"Parsing files from {self.part} part..."
            ):
                name = os.path.basename(fn)
                name, ext = os.path.splitext(name)
                if ext == ".txt":
                    with open(fn, encoding='utf-8') as file_obj:
                        text = file_obj.read()
                    self.texts[name] = text
                elif ext == ".ann":
                    with open(fn, encoding='utf-8') as file_obj:
                        ann = file_obj.read().strip().split('\n')
                    self.anns[name] = ann
                elif ext == ".norm":
                    with open(fn, encoding='utf-8') as file_obj:
                        norm = file_obj.read().strip().split('\n')
                    self.norms[name] = norm

    def _make_raw_data(self, is_save=True):
        file_list = None
        if is_save:
            file_list = open(os.path.join(self.output_dir, self.file_list_name), "w")
        for name in tqdm(self.texts, total=len(self.texts), leave=False, desc="Making raw files..."):
            text = self.texts[name]
            if self.answer_sep in text:
                print("answer_sep in file", name)
            ann = self.anns[name]
            if self.part == self.train_part_name:
                norm = self.norms[name]
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
                final_text = final_text.format(
                    bos=self.tokenizer.bos_token,
                    text=text[:stop],
                    answ_sep=self.answer_sep,
                    ln=line_norm,
                    eos=self.tokenizer.eos_token,
                )
                file_texts.append(final_text)
            self.lm_prefixes[name] = file_texts

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
        default=" A: ",
        help='separator between query and answer.'
    )
    return parser


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Prepare data for language modeling")
    arg_parser = add_data_reader_arguments(arg_parser)
    args = arg_parser.parse_args()
    args = vars(args)
    args["data_parts"] = args["data_parts"].split(",")
    reader = DataReader(**args)
    reader()
