import argparse


def get_arg_parser_baseline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--label_level", type=str)
    return parser

def get_arg_parser_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--label_level", type=str)
    parser.add_argument("--category", type=str, required=False)
    return parser

def get_arg_parser_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--classifier_path", type=str)
    parser.add_argument("--generative_path", type=str, required=False)
    parser.add_argument("--label_level", type=str)
    return parser

def get_arg_parser_test_and_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--classifier_path", type=str)
    parser.add_argument("--generative_path", type=str, required=False)
    parser.add_argument("--label_level", type=str)
    parser.add_argument("--documents_interval", type=int)
    parser.add_argument("--training_size", type=int)
    return parser

def get_arg_parser_with_evolution():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_path", type=str)
    return parser


def get_arg_parser_with_evolution_with_threshold():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--threshold", type=float)
    return parser