import os
from conllu.parser import (
    ParseException,
    DEFAULT_FIELDS,
    DEFAULT_FIELD_PARSERS,
    parse_line
)
from random import randint
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from modules.utils import get_all_files_from_dir


def parse_token_and_metadata(data, fields=None, field_parsers=None):
    if not data:
        raise ParseException("Can't create TokenList, no data sent to constructor.")

    fields = fields or DEFAULT_FIELDS
    field_parsers = field_parsers or DEFAULT_FIELD_PARSERS

    tokens = []
    texts = []

    for line in data.split('\n'):
        line = line.strip()

        if not line:
            continue

        if line.startswith('#'):
            var_name, var_value = parse_comment_line(line)
            if var_name == "text":
                texts.append(var_value)
        else:
            tokens.append(parse_line(line, fields, field_parsers))

    return tokens, texts


def parse_comment_line(line):
    line = line.strip()

    if line[0] != '#':
        raise ParseException("Invalid comment format, comment must start with '#'")

    stripped = line[1:].strip()
    if '=' not in line and stripped != 'newdoc' and stripped != 'newpar':
        return None, None

    name_value = line[1:].split('=', 1)
    var_name = name_value[0].strip()
    var_value = None if len(name_value) == 1 else name_value[1].strip()

    return var_name, var_value


def add_data_reader_arguments(parser):
    group = parser.add_argument_group('parse_conllu', 'Prepare data at conllu format for reader.')
    group.add_argument(
        '--path',
        type=str,
        help='path to the data dir'
    )
    group.add_argument(
        '--output_dir',
        type=str,
        help='path to the data dir'
    )
    group.add_argument(
        '--num_tries',
        type=int,
        default=100,
        help='Number of tries for random generation.'
    )
    group.add_argument(
        '--num_proc',
        type=int,
        default=4,
        help='Number of processes.'
    )
    group.add_argument(
        '--test_size',
        type=float,
        default=0.1,
        help='Size of norms for generation.'
    )
    return parser


def main():
    arg_parser = argparse.ArgumentParser(description="Prepare conllu data for reader")
    arg_parser = add_data_reader_arguments(arg_parser)
    args = arg_parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    def worker(path):
        with open(path, "r") as file:
            tokens, texts = parse_token_and_metadata(file.read())

        text = " ".join(texts).replace("\n", " ").strip()
        anns = []
        norms = []
        tok_len = len(tokens)
        count = int(tok_len * args.test_size)
        cur_try = 0
        while len(anns) != count and cur_try < args.num_tries:
            idx = randint(0, tok_len - 1)
            cur_try += 1
            # print(idx, tok_len, len(tokens))
            if tokens[idx]["upos"] not in ["PUNCT", "ADP", "PART"]:
                if tokens[idx]["lemma"]:
                    norm = tokens[idx]["lemma"]
                    token = tokens[idx]["form"]
                    start = text.find(token)
                    if -1 < start:
                        anns.append(f"{start} {start + len(token)}")
                        norms.append(norm)
                    else:
                        print(text, token)
        bn = os.path.basename(path)[-4:]
        with open(os.path.join(args.output_dir, f"{bn}.txt"), "w") as file:
            file.write(text)
        with open(os.path.join(args.output_dir, f"{bn}.norm"), "w") as file:
            file.write("\n".join(norms))
        with open(os.path.join(args.output_dir, f"{bn}.ann"), "w") as file:
            file.write("\n".join(anns))

    print("Start running on ", args.num_proc, "processes")
    files = [x for x in get_all_files_from_dir(args.path) if x.endswith(".txt")]
    with ProcessPoolExecutor(args.num_proc) as pool:
        res = pool.map(worker, files, chunksize=1)
        list(tqdm(res, total=len(files)))
    print("Done!")


if __name__ == "__main__":
    main()
