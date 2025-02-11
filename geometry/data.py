import sae_lens
import torch
import datasets
from typing import List, Tuple
from tqdm import tqdm


def fetch_sae_and_model(
    sae_name: str,
) -> Tuple[List[sae_lens.SAE], sae_lens.HookedSAETransformer]:
    """ fetch the specified SAE and model given the SAE name

    Args:
        sae_name (str): the name of the SAE

    Returns:
        Tuple[List[sae_lens.SAE], sae_lens.HookedSAETransformer]: the SAEs and the model
    """    
    if sae_name == "llama3.1-8b":
        model_name = "meta-llama/Llama-3.1-8B"
        layers = 32
        release = "llama_scope_lxr_8x"
    elif sae_name == "pythia-70m-deduped":
        model_name = "EleutherAI/pythia-70m-deduped"
        layers = 6
        release = "pythia-70m-deduped-res-sm"
    elif sae_name == "gemma-2-2b":
        model_name = "gemma-2-2b"
        layers = 26
        release = "gemma-scope-2b-pt-res-canonical"
    saes = fetch_sae(release, layers)
    model = fetch_model(model_name)
    return saes, model


def fetch_sae(release: str, layers: int) -> sae_lens.SAE:
    saes = []
    for layer in tqdm(range(layers)):
        if release == "gemma-scope-2b-pt-res-canonical":
            sae_id = f"layer_{layer}/width_16k/canonical"
        elif release == "llama_scope_lxr_8x":
            sae_id = f"l{layer}r_8x"
        elif release == "pythia-70m-deduped-res-sm":
            sae_id = f"blocks.{layer}.hook_resid_post"
        sae = sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0]
        sae.to(dtype=torch.bfloat16)
        saes.append(sae)
    return saes


def fetch_model(model_name: str) -> sae_lens.HookedSAETransformer:
    model = sae_lens.HookedSAETransformer.from_pretrained(
        model_name, dtype=torch.bfloat16
    )
    return model


def load_dataset_by_name(dataset_name: str) -> Tuple[datasets.Dataset, float, str]:
    """Load the specified dataset

    Args:
        dataset_name (str): the name of the dataset

    Returns:
        Tuple[datasets.Dataset, float, str]:
            the dataset, the ratio of the dataset to use, the text column name
    """
    if dataset_name == "wikitext":
        dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "test"
        ]
        ratio = 1e-2
        text = "text"
    elif dataset_name == "sql-create-context":
        dataset = datasets.load_dataset("b-mc2/sql-create-context")["train"]
        ratio = 5e-2
        text = "answer"
    elif dataset_name == "MathInstruct":
        dataset = datasets.load_dataset("TIGER-Lab/MathInstruct")["train"]
        text = "output"
        ratio = 1e-2
    elif dataset_name == "MMLU":
        dataset = datasets.load_dataset("cais/mmlu", "all")['validation']
        ratio = 1e-2
        text = 'question'
    return dataset, ratio, text
