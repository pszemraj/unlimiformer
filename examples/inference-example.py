import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import fire
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from unlimiformer import Unlimiformer, UnlimiformerArguments


def load_and_prepare_models(
    modelname: str, tokenizer_name: str = "facebook/bart-base"
) -> tuple:
    if tokenizer_name is None:
        tokenizer_name = modelname
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def process_example(
    tokenizer, example_input: str, max_input_length: int = 1024, device: str = "cpu"
) -> tuple:
    """
    process_example - process an example input for inference & comparison

    :param AutoTokenizer tokenizer: tokenizer to use
    :param str example_input: text input to process
    :param int max_input_length: maximum input length for the truncated example, defaults to 1024
    :param str device: device to use, defaults to "cpu"
    :return tuple: tuple of example tokens and truncated example tokens
    """
    example_tokens = tokenizer(example_input, truncation=False, return_tensors="pt")
    truncated_example_tokens = tokenizer(
        example_input, truncation=True, max_length=max_input_length, return_tensors="pt"
    )
    example_tokens.to(device)
    truncated_example_tokens.to(device)

    return example_tokens, truncated_example_tokens


def prepare_unlimiformer_kwargs(
    tokenizer, defaults: UnlimiformerArguments = None
) -> dict:
    """prepare_unlimiformer_kwargs - prepare the unlimiformer kwarg (here, use the defaults)"""
    defaults = UnlimiformerArguments() if defaults is None else defaults
    unlimiformer_kwargs = {
        "layer_begin": defaults.layer_begin,
        "layer_end": defaults.layer_end,
        "unlimiformer_head_num": defaults.unlimiformer_head_num,
        "exclude_attention": defaults.unlimiformer_exclude,
        "chunk_overlap": defaults.unlimiformer_chunk_overlap,
        "model_encoder_max_len": defaults.unlimiformer_chunk_size,
        "verbose": defaults.unlimiformer_verbose,
        "unlimiformer_training": defaults.unlimiformer_training,
        "use_datastore": defaults.use_datastore,
        "flat_index": defaults.flat_index,
        "test_datastore": defaults.test_datastore,
        "reconstruct_embeddings": defaults.reconstruct_embeddings,
        "gpu_datastore": defaults.gpu_datastore,
        "gpu_index": defaults.gpu_index,
    }
    unlimiformer_kwargs["tokenizer"] = tokenizer
    return unlimiformer_kwargs


def generate_output(model, example, tokenizer, max_new_tokens: int = 512) -> str:
    return tokenizer.batch_decode(
        model.generate(**example, max_new_tokens=max_new_tokens),
        ignore_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]


def run_inference(
    model_name: str = "abertsch/unlimiformer-bart-govreport-alternating",
    tokenizer_name: str = "facebook/bart-base",
    dataset_name: str = "urialon/gov_report_validation",
    split_name: str = "validation",
    input_column: str = "input",
    example_index: int = 0,
    max_length: int = 1024,
    max_new_tokens: int = 512,
) -> None:
    """
    run_inference - Basic example of running inference with Unlimiformer

    :param str model_name: name of the Unlimiformer model to use, defaults to "abertsch/unlimiformer-bart-govreport-alternating"
    :param str tokenizer_name: model/repo name of the tokenizer to use, defaults to "facebook/bart-base"
    :param str dataset_name: name of the dataset to use, defaults to "urialon/gov_report_validation"
    :param str split_name: name of the dataset split to use, defaults to "validation"
    :param str input_column: name of the column in the dataset to use as input, defaults to "input"
    :param int example_index: index of the example to use, defaults to 0
    :param int max_length: maximum length of the input (for the base model), defaults to 1024 (BART-base max)
    :param int max_new_tokens: maximum number of tokens to generate, defaults to 512
    """
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset_name)

    model, tokenizer = load_and_prepare_models(model_name, tokenizer_name)
    example_input = dataset[split_name][example_index][input_column]
    example, truncated_example = process_example(
        tokenizer, example_input, max_length, device
    )

    print(f"INPUT LENGTH (tokens): {example['input_ids'].shape[-1]}")

    model.to(device)
    truncated_out = generate_output(
        model, truncated_example, tokenizer, max_new_tokens=max_new_tokens
    )
    print(f"Standard BART output:\t{truncated_out}", "\n" * 2)
    # setup model
    defaults = UnlimiformerArguments()
    unlimiformer_kwargs = prepare_unlimiformer_kwargs(
        tokenizer=tokenizer, defaults=defaults
    )
    model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)
    model = model.to(device)
    model.eval()

    unlimiformer_out = generate_output(
        model, example, tokenizer, max_new_tokens=max_new_tokens
    )
    print(f"Unlimiformer output:\t{unlimiformer_out}")


if __name__ == "__main__":
    fire.Fire(run_inference)
