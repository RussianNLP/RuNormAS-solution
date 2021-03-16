from modules.data.read import DataReader, add_data_reader_arguments
import argparse
from predict_runormas import predict
from src.xl_wrapper import RuGPT3XL
from src.utils import print_args


def get_model(reader, args):
    model = RuGPT3XL.from_pretrained(
        model_name_or_path=args.tokenizer_name,
        seq_len=512,
        weights_path=args.weights_path,
        deepspeed_config_path=args.deepspeed_config_path
    )
    model.tokenizer = reader.tokenizer
    return model


def add_prediction_arguments(parser):
    group = parser.add_argument_group('prediction', 'Prediction arguments')
    group.add_argument(
        '--weights_path',
        type=str,
        help='path pretrained model'
    )
    group.add_argument(
        '--deepspeed_config_path',
        type=str,
        default="",
        help='path to deepspeed config'
    )
    return parser


def main():
    arg_parser = argparse.ArgumentParser(description="Predict")
    arg_parser = add_data_reader_arguments(arg_parser)
    arg_parser = add_prediction_arguments(arg_parser)
    args = arg_parser.parse_args()
    args.data_parts = args.data_parts.split(",")
    print_args(args)
    reader = DataReader(**vars(args))
    reader.prc(is_save=False)
    model = get_model(reader, args)
    predict(reader, model, args.save_preds_path, 0, num_beams=args.num_beams, do_sample=bool(args.do_sample))


if __name__ == "__main__":
    main()
