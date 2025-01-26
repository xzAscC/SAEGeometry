import sae_lens
import torch
import datasets
import argparse
import random
import numpy as np
from torch import autocast
from typing import List, Tuple
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    """Set the seed for reproducibility.

    Args:
        seed: The seed value.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="The seed value.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="pythia",
        help="The model name.",
        choices=["gemma2", "llama3", "pythia"],
    )
    args = parser.parse_args()
    return args


def obtain_pythia_data(
    layers: int = 6,
) -> Tuple[List[sae_lens.SAE], sae_lens.HookedSAETransformer]:
    """
    Obtain the data.
    """

    release = "pythia-70m-deduped-res-sm"

    model_name = "EleutherAI/pythia-70m-deduped"
    saes = []
    for layer in tqdm(range(layers)):
        sae_id = f"blocks.{layer}.hook_resid_post"
        sae = sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0]
        saes.append(sae)
    model = sae_lens.HookedSAETransformer.from_pretrained(model_name)
    return saes, model


def obtain_llama_data(
    layers: int = 32,
) -> Tuple[List[sae_lens.SAE], sae_lens.HookedSAETransformer]:
    """
    Obtain the data.
    """
    model_name = "meta-llama/Llama-3.1-8B"
    saes = []
    release = "llama_scope_lxr_8x"
    for layer in tqdm(range(layers)):
        sae_id = f"l{layer}r_8x"
        sae = sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0]
        sae.to(dtype=torch.bfloat16)
        saes.append(sae)
    model = sae_lens.HookedSAETransformer.from_pretrained(
        model_name, dtype=torch.bfloat16
    )
    return saes, model


def obtain_gemma_data(
    layers: int = 26,
) -> Tuple[List[sae_lens.SAE], sae_lens.HookedSAETransformer]:
    """
    Obtain the data.
    """
    model_name = "gemma-2-2b"
    saes = []
    release = "gemma-scope-2b-pt-res-canonical"
    for layer in tqdm(range(layers)):
        sae_id = f"layer_{layer}/width_16k/canonical"
        sae = sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0]
        sae.to(dtype=torch.bfloat16)
        saes.append(sae)
    model = sae_lens.HookedSAETransformer.from_pretrained(
        model_name, dtype=torch.bfloat16
    )
    return saes, model


@torch.no_grad()
def obtain_acts(
    saes, model, layers: int = 26, model_name: str = "gemma"
) -> torch.Tensor:
    """
    Obtain activations from the model.
    """
    ds = ["wiki", "code", "math"]
    # ds = ["math"]
    for idx in range(len(ds)):
        data_name = ds[idx]
        if data_name == "wiki":
            dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
                "train"
            ]
            text = "text"
            if model_name == "gemma2":
                ds_ratio = 0.5
            elif model_name == "llama3":
                ds_ratio = 0.05
            else:
                ds_ratio = 1
        elif data_name == "code":
            dataset = datasets.load_dataset("b-mc2/sql-create-context")["train"]
            text = "answer"
            if model_name == "gemma2":
                ds_ratio = 0.5
            elif model_name == "llama3":
                ds_ratio = 0.05
            else:
                ds_ratio = 1
        elif data_name == "math":
            dataset = datasets.load_dataset("TIGER-Lab/MathInstruct")["train"]
            text = "output"
            if model_name == "gemma2":
                ds_ratio = 0.1
            elif model_name == "llama3":
                ds_ratio = 0.01
            else:
                ds_ratio = 1
        freqs = torch.zeros(layers, saes[0].cfg.d_sae)
        doc_len = 0
        length_ds = int(len(dataset) * ds_ratio)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            for idx in tqdm(range(length_ds)):
                example = dataset[idx]
                tokens = model.to_tokens([example[text]], prepend_bos=True)
                _, cache1 = model.run_with_cache_with_saes(
                    tokens, saes=saes, use_error_term=True
                )
                local_doc_len = cache1[
                    f"blocks.0.hook_resid_post.hook_sae_acts_post"
                ].shape[1]
                freq = torch.zeros_like(freqs)
                new_doc_len = doc_len + local_doc_len
                for layer in range(layers):
                    prompt2 = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
                    freq[layer] = (
                        (cache1[prompt2] > 1e-3)[0].sum(0) / local_doc_len
                    ).cpu()
                if idx == 0:
                    freqs = freq
                else:
                    freqs = (
                        freqs * doc_len / new_doc_len
                        + freq * local_doc_len / new_doc_len
                    )
                doc_len = new_doc_len
        torch.save(freqs, f"{model_name}_freqs_{data_name}.pt")
    return freqs


if __name__ == "__main__":
    args = config()
    set_seed(args.seed)
    if args.model_name == "gemma2":
        saes, model = obtain_gemma_data()
        layers = 26
    elif args.model_name == "llama3":
        saes, model = obtain_llama_data()
        layers = 32
    else:
        saes, model = obtain_pythia_data()
        layers = 6
    # saes, model = obtain_gemma_data()
    # saes, model = obtain_llama_data()
    torch.cuda.empty_cache()
    acts = obtain_acts(saes, model, layers, model_name=args.model_name)
