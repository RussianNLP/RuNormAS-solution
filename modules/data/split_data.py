from sklearn.model_selection import train_test_split
import argparse


def add_data_split_arguments(parser):
    group = parser.add_argument_group('data_split', 'Data train path.')
    group.add_argument(
        '--path',
        default="../data/train_new_raw/files.list",
        type=str,
        help='path to the train files.list'
    )
    group.add_argument(
        '--train',
        type=str,
        default="../data/train_new_raw/files_train.list",
        help='path to store actual train files.list'
    )
    group.add_argument(
        '--valid',
        type=str,
        default="../data/train_new_raw/files_valid.list",
        help='path to store valid files.list'
    )
    group.add_argument(
        '--test_size',
        type=float,
        default=0.01,
        help='valid size'
    )
    return parser


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Split train data to train and valid")
    arg_parser = add_data_split_arguments(arg_parser)
    args = arg_parser.parse_args()

    with open(args.path, "r") as file:
        fns = file.read().strip().split("\n")
    train, valid = train_test_split(fns, test_size=args.test_size)
    with open(args.train, "w") as file:
        file.write("\n".join(train))

    with open(args.valid, "w") as file:
        file.write("\n".join(valid))
