from tqdm import tqdm
import os
from src.utils import get_all_files_from_dir
from transformers import GPT2Tokenizer


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

    def _read_files(self):
        for data_part in self.data_parts:
            all_files = get_all_files_from_dir(os.path.join(self.path, data_part))
            for fn in tqdm(
                    all_files, total=len(all_files), leave=False,
                    desc=f"Parsing files from {self.part} part..."
            ):
                name, ext = os.path.basename(fn)
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

    def _make_raw_data(self):
        with open(os.path.join(self.output_dir, self.file_list_name), "w") as file_list:
            for fn in self.texts:
                text = self.texts[fn]
                if self.answer_sep in text:
                    print("answer_sep in file", fn)
                ann = self.anns[fn]
                if self.part == self.train_part_name:
                    norm = self.norms[fn]
                    if len(norm) != len(ann):
                        raise ValueError(f"Lens of norm and ann should be same. Error with file name {fn}")
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

                res_fn = os.path.join(self.output_dir, self.files_dir_name, f"{fn}.txt")
                file_list.write(f"{res_fn}\n")

                final_file_text = "\n".join(file_texts)
                with open(res_fn, "w", encoding='utf-8') as file_out:
                    file_out.write(final_file_text)
                self.texts_for_model.append(final_file_text)
