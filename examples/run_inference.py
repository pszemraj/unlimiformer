import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import fire
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from unlimiformer import Unlimiformer, UnlimiformerArguments

from constants import PRETRAINED_MODELS


def get_model_and_tokenizer_name(identifier: str) -> tuple:
    if identifier in PRETRAINED_MODELS:
        return (
            PRETRAINED_MODELS[identifier]["model_name"],
            PRETRAINED_MODELS[identifier]["tokenizer_name"],
        )
    else:
        for key, model_info in PRETRAINED_MODELS.items():
            if model_info["model_name"] == identifier:
                return model_info["model_name"], model_info["tokenizer_name"]
    raise ValueError(
        f"Identifier {identifier} not found in the pretrained models dictionary."
    )


def load_and_prepare_models(
    modelname: str, tokenizer_name: str = "facebook/bart-base"
) -> tuple:
    if tokenizer_name is None:
        tokenizer_name = modelname
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


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


def run_inference(
    model_name: str = "abertsch/unlimiformer-bart-govreport-alternating",
    tokenizer_name: str = None,
    dataset_name: str = None,
    split_name: str = "validation",
    source_column: str = "input",
    max_input_length: int = 16384,
    max_new_tokens: int = 1024,
    input_path: str = None,
    recursive: bool = False,
    max_samples: int = None,
    output_dir: str = None,
    compile_model: bool = False,
    debug: bool = False,
    **generate_kwargs,
) -> None:
    if dataset_name is None and input_path is None:
        raise ValueError(
            "One of dataset_name, input_file, or input_dir must be provided."
        )

    logger = logging.getLogger(__name__)
    if debug:
        logger.setLevel(logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer_name is None:
        model_name, tokenizer_name = get_model_and_tokenizer_name(model_name)
        logger.info(f"Using tokenizer {tokenizer_name} from model {model_name}")

    model, tokenizer = load_and_prepare_models(model_name, tokenizer_name)

    if dataset_name is not None:
        logger.info(f"Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_name)
        input_texts = [x[source_column] for x in dataset[split_name]]
    else:
        input_path = Path(input_path)
        logger.info(f"Loading input from {input_path}...")
        if input_path.is_dir():
            logger.info(
                f"Loading input from {input_path} as directory. Recursive: {recursive}"
            )
            input_texts = []
            for filepath in (
                input_path.rglob("*") if recursive else input_path.glob("*")
            ):
                if filepath.suffix == ".txt":
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                        input_texts.append(file.read())
            logger.info(f"Loaded {len(input_texts)} files from {input_path}")
        elif input_path.is_file():
            with open(input_path, "r", encoding="utf-8", errors="ignore") as file:
                input_texts = [file.read()]

    if max_samples is not None:
        logger.info(f"Limiting to {max_samples} samples")
        input_texts = input_texts[:max_samples]

    if output_dir is None:
        output_dir = (
            input_path.parent / f"{input_path.stem}_unlimiformer_summaries"
            if input_path is not None
            else Path.cwd() / "unlimiformer_summaries"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving summaries to {output_dir}")

    # setup model
    defaults = UnlimiformerArguments()
    unlimiformer_kwargs = prepare_unlimiformer_kwargs(
        tokenizer=tokenizer, defaults=defaults
    )
    model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)
    model = model.to(device)
    model.eval()

    if compile_model:
        logger.info("Compiling model...")
        model = torch.compile(model)

    logger.info("Running inference...")
    for i, input_text in tqdm(enumerate(input_texts), desc="Inference"):
        logger.debug(f"Processing input {i+1}/{len(input_texts)}...")
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        ).to(model.device)
        summary_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, **generate_kwargs
        )
        summary = tokenizer.decode(
            summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        output_file = output_dir / f"summary_{i+1}.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(summary)

        logger.debug(
            f"Processed input {i+1}/{len(input_texts)} and saved summary to {output_file}"
        )
    logger.info(
        f"Finished inference. Summaries saved to:\n\t{str(output_dir.resolve())}"
    )


if __name__ == "__main__":
    fire.Fire(run_inference)
