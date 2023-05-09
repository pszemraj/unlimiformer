import json
import sys
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
    input_data: str,
    model_name: str = "abertsch/unlimiformer-bart-booksum-alternating",
    tokenizer_name: str = None,
    split_name: str = "validation",
    source_column: str = "input",
    max_input_length: int = 32768,
    max_new_tokens: int = 1024,
    num_beams: int = 4,
    recursive: bool = False,
    max_samples: int = None,
    output_dir: str = None,
    compile_model: bool = False,
    debug: bool = False,
    **generate_kwargs,
) -> None:
    """
    Run inference with the Unlimiformer model on a dataset or input files and save the summaries to a specified directory.

    Args:
        input_data (str, optional): Source data to be summarized. Can be a path to a file or directory, or a dataset name.
        model_name (str, optional): Name or key of the pretrained model.
        tokenizer_name (str, optional): Name of the tokenizer to be used with the model.
        split_name (str, optional): Dataset split to be used (e.g., "validation").
        source_column (str, optional): Column in the dataset containing the input text.
        max_input_length (int, optional): Maximum input length for the tokenizer.
        max_new_tokens (int, optional): Maximum number of tokens in the generated summary.
        num_beams (int, optional): Number of beams to use for beam search.
        recursive (bool, optional): Whether to search for input files recursively in the input directory.
        max_samples (int, optional): Maximum number of samples to process.
        output_dir (str, optional): Directory where the summaries will be saved.
        compile_model (bool, optional): Whether to compile the model before running inference.
        debug (bool, optional): Whether to enable debug logging.
        **generate_kwargs: Additional keyword arguments to be passed to the generate method of the model.

    Returns:
        None
    """

    logger = logging.getLogger(__name__)
    if debug:
        logger.setLevel(logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer_name is None:
        model_name, tokenizer_name = get_model_and_tokenizer_name(model_name)
        logger.info(f"Using tokenizer {tokenizer_name} from model {model_name}")

    model, tokenizer = load_and_prepare_models(model_name, tokenizer_name)

    if Path(input_data).exists():
        logger.info(f"Loading input from {input_data}...")
        input_data = Path(input_data)

        if input_data.is_dir():
            logger.info(
                f"Loading input from {input_data} as directory. Recursive: {recursive}"
            )
            input_texts = {}
            for filepath in (
                input_data.rglob("*") if recursive else input_data.glob("*")
            ):
                if filepath.suffix == ".txt":
                    try:
                        with open(
                            filepath, "r", encoding="utf-8", errors="ignore"
                        ) as file:
                            input_texts[filepath.stem] = file.read()
                    except Exception as e:
                        logger.error(f"Failed to read file {filepath}: {e}")
            logger.info(f"Loaded {len(input_texts)} files from {input_data}")
        elif input_data.is_file():
            try:
                with open(input_data, "r", encoding="utf-8", errors="ignore") as file:
                    input_texts = {input_data.stem: file.read()}
            except Exception as e:
                logger.error(f"Failed to read file {input_data}: {e}")
                sys.exit(1)
    else:
        logger.info(f"Loading dataset {input_data}...")
        dataset = load_dataset(input_data)
        input_texts = {
            f"{input_data}_{split_name}_{i}": x[source_column]
            for i, x in enumerate(dataset[split_name])
        }

    if max_samples is not None:
        logger.info(f"Limiting to {max_samples} samples")
        input_texts = dict(list(input_texts.items())[:max_samples])

    if output_dir is None:
        output_dir = (
            input_data.parent / f"{input_data.stem}_unlimiformer_summaries"
            if Path(input_data).exists()
            else Path.cwd() / "unlimiformer-summaries"
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
    for i, (semantic_label, input_text) in tqdm(
        enumerate(input_texts.items()), desc="Inference", total=len(input_texts)
    ):
        logger.debug(f"Processing input {i+1}/{len(input_texts)}...")
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        ).to(model.device)
        summary_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            **generate_kwargs,
        )
        summary = tokenizer.decode(
            summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        output_file = output_dir / f"summary_{i}_{semantic_label}.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(summary)
            logger.debug(
                f"Processed input {i+1}/{len(input_texts)} and saved summary to {output_file}"
            )
        except Exception as e:
            logger.error(f"Failed to write summary to {output_file}: {e}")

    # Save settings to a JSON file
    settings = {
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "input_data": str(input_data),
        "split_name": split_name,
        "source_column": source_column,
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "early_stopping": True,
        "recursive": recursive,
        "max_samples": max_samples,
        "output_dir": str(output_dir),
        "compile_model": compile_model,
        "generate_kwargs": generate_kwargs,
    }
    settings_file = output_dir / "summarization_parameters.json"
    with open(settings_file, "w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=False, indent=4)

    logger.info(f"Settings saved to {settings_file}")
    logger.info(
        f"Finished inference. Summaries saved to:\n\t{str(output_dir.resolve())}"
    )


if __name__ == "__main__":
    fire.Fire(run_inference)
