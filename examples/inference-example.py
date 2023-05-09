import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import fire
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BartForConditionalGeneration

from unlimiformer import Unlimiformer, UnlimiformerArguments


def load_and_prepare_models(
    modelname: str, tokenizer_name: str = "facebook/bart-base"
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = BartForConditionalGeneration.from_pretrained(modelname)
    return tokenizer, model


def process_example(
    tokenizer, example_input: str, max_input_length: int = 1024, device: str = "cpu"
) -> tuple:
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
    )[0]


def run_inference(
    modelname: str = "abertsch/unlimiformer-bart-govreport-alternating",
    tokenizer_name: str = "facebook/bart-base",
    dataset_name: str = "urialon/gov_report_validation",
    split_name: str = "validation",
    source_column: str = "input",
    max_length: int = 1024,
    max_new_tokens: int = 512,
) -> None:
    """
    run_inference -  basic inference example

    :param str modelname: _description_, defaults to "abertsch/unlimiformer-bart-govreport-alternating"
    :param str tokenizer_name: _description_, defaults to "facebook/bart-base"
    :param str dataset_name: _description_, defaults to "urialon/gov_report_validation"
    :param str split_name: _description_, defaults to "validation"
    :param str source_column: _description_, defaults to "input"
    :param int max_length: _description_, defaults to 1024
    :param int max_new_tokens: _description_, defaults to 512
    """
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset_name)

    tokenizer, model = load_and_prepare_models(modelname, tokenizer_name)
    example_input = dataset[split_name][0][source_column]
    example, truncated_example = process_example(
        tokenizer, example_input, max_length, device
    )

    print(f"INPUT LENGTH (tokens): {example['input_ids'].shape[-1]}")

    model.to(device)
    truncated_out = generate_output(
        model, truncated_example, tokenizer, max_new_tokens=max_new_tokens
    )
    print(f"Standard BART output:\t{truncated_out}")
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
