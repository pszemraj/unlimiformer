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
    tokenizer, example_input: str, max_length: int, device: torch.device
) -> tuple:
    example = tokenizer(example_input, truncation=False, return_tensors="pt")
    truncated_example = tokenizer(
        example_input, truncation=True, max_length=max_length, return_tensors="pt"
    )
    example.to(device)
    truncated_example.to(device)

    return example, truncated_example


def prepare_unlimiformer_kwargs(defaults: UnlimiformerArguments, tokenizer) -> dict:
    unlimiformer_kwargs = {
        key: getattr(defaults, key)
        for key in dir(defaults)
        if not key.startswith("__") and not callable(getattr(defaults, key))
    }
    unlimiformer_kwargs["tokenizer"] = tokenizer
    return unlimiformer_kwargs


def generate_output(model, example, max_length: int, tokenizer) -> str:
    return tokenizer.batch_decode(
        model.generate(**example, max_length=max_length), ignore_special_tokens=True
    )[0]


def run_inference(
    modelname: str = "abertsch/unlimiformer-bart-govreport-alternating",
    tokenizer_name: str = "facebook/bart-base",
    dataset_name: str = "urialon/gov_report_validation",
    split_name: str = "validation",
    max_length: int = 1024,
) -> None:
    """
    run_inference -  basic inference example

    :param str modelname: _description_, defaults to "abertsch/unlimiformer-bart-govreport-alternating"
    :param str tokenizer_name: _description_, defaults to "facebook/bart-base"
    :param str dataset_name: _description_, defaults to "urialon/gov_report_validation"
    :param str split_name: _description_, defaults to "validation"
    :param int max_length: _description_, defaults to 1024
    """
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset_name)

    tokenizer, model = load_and_prepare_models(modelname, tokenizer_name)
    example_input = dataset["validation"][0]["input"]
    example, truncated_example = process_example(
        tokenizer, example_input, max_length, device
    )

    print(f"INPUT LENGTH (tokens): {example['input_ids'].shape[-1]}")

    defaults = UnlimiformerArguments()
    unlimiformer_kwargs = prepare_unlimiformer_kwargs(defaults, tokenizer)

    model.to(device)
    truncated_out = generate_output(model, truncated_example, max_length, tokenizer)

    model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)
    model.eval()
    model.to(device)

    unlimiformer_out = generate_output(model, example, max_length, tokenizer)
    print(unlimiformer_out)


if __name__ == "__main__":
    fire.Fire(run_inference)
