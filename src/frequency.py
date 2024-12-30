import torch
import argparse
import sae_lens
import transformer_lens
import datasets
import os
import jaxtyping
import pandas as pd
import plotly.express as px
import plotly.colors as pc
from tqdm import tqdm
from typing import Tuple
from utils import set_seed, get_device
from logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed value")
    parser.add_argument(
        "--sae_name",
        type=str,
        default="pythia-70-res",
        help="SAE model name",
        choices=["pythia-70-res", "gpt2-small-res", "get2-medium-res"],
    )
    parser.add_argument(
        "--activation_path",
        type=str,
        default="./res/freq_mean_activation",
        help="Path to save the activation",
    )
    return parser.parse_args()


@torch.no_grad()
def load_sae_from_saelens(
    sae_name: str, device: str
) -> Tuple[torch.nn.Module, sae_lens.HookedSAETransformer, torch.utils.data.Dataset]:
    sae_list = []
    match sae_name:
        case "pythia-70-res":
            layers = 6
            release = "pythia-70m-deduped-res-sm"
            sae_id = f"blocks.0.hook_resid_pre"
            model_name = "pythia-70m-deduped"
            sae_list.append(
                sae_lens.SAE.from_pretrained(
                    release=release, sae_id=sae_id, device=device
                )[0]
            )
            for layer in range(layers):
                sae_id = f"blocks.{layer}.hook_resid_post"
                sae_list.append(
                    sae_lens.SAE.from_pretrained(
                        release=release, sae_id=sae_id, device=device
                    )[0]
                )

            model = sae_lens.HookedSAETransformer.from_pretrained(model_name).to(device)
            # TODO: add different datasets
            dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
                "train"
            ]
        case "gpt2-small-res":
            pass
        case "get2-medium-res":
            pass
        case _:
            raise ValueError(f"Invalid SAE model name: {sae_name}")

    return sae_list, model, dataset


def obtain_activations(
    sae_list: torch.nn.Module,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    activation_path: str = None,
) -> torch.Tensor:
    "output: (num_layers, num_features)"
    if activation_path:
        if os.path.exists("freq_mean_global.pt"):
            return torch.load("freq_mean_global.pt", weights_only=True)
        else:
            raise ValueError("freq_mean_global.pt not found")
    match sae_list[0].cfg.model_name:
        case "pythia-70m-deduped":
            doc_len = 0
            freq_mean_global = 0
            layers = 6
            freqs = torch.zeros(layers + 1, sae_list[0].cfg.d_sae).to(device)
            for idx in tqdm(range(len(dataset))):
                # loop begin, fuck indent
                example = dataset[idx]
                tokens = model.to_tokens([example["text"]], prepend_bos=True)
                _, cache = model.run_with_cache_with_saes(tokens, saes=sae_list)
                local_doc_len = cache[
                    "blocks.0.hook_resid_post.hook_sae_acts_post"
                ].shape[1]
                freq = torch.zeros_like(freqs)
                for layer in range(layers):
                    prompt = f"blocks.{layer}.hook_resid_pre.hook_sae_acts_post"
                    prompt2 = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
                    if layer == 0:
                        freq[layer] = (cache[prompt] > 1e-3)[0].sum(0) / local_doc_len
                    else:
                        freq[layer + 1] = (cache[prompt2] > 1e-3)[0].sum(
                            0
                        ) / local_doc_len
                new_doc_len = doc_len + local_doc_len
                if idx == 0:
                    freq_mean_global = freq
                else:
                    freq_mean_global = (
                        freq_mean_global * doc_len / new_doc_len
                        + freq * local_doc_len / new_doc_len
                    )
                doc_len = new_doc_len
                # loop end
        case "gpt2":
            pass
        case _:
            raise ValueError(f"Invalid model name: {sae_list[0].cfg.model_name}")

    if os.path.exists("freq_mean_global.pt"):
        os.remove("freq_mean_global.pt")
    torch.save(freq_mean_global, "freq_mean_global.pt")
    return freq_mean_global


@torch.no_grad()
def get_cosine_similarity(
    dict_elements_1: jaxtyping.Float[torch.Tensor, "d_sae d_llm"],
    dict_elements_2: jaxtyping.Float[torch.Tensor, "d_sae d_llm"],
    p: int = 2,
    dim: int = 1,
    normalized: bool = True,
) -> jaxtyping.Float[torch.Tensor, "d_llm d_llm"]:
    """Get the cosine similarity between the dictionary elements.

    Args:
        dict_elements_1: The first dictionary elements.
        dict_elements_2: The second dictionary elements.

    Returns:
        The cosine similarity between the dictionary elements.
    """
    # Compute cosine similarity in pytorch
    dict_elements_1 = dict_elements_1
    dict_elements_2 = dict_elements_2

    # Normalize the tensors
    if normalized:
        dict_elements_1 = torch.nn.functional.normalize(dict_elements_1, p=p, dim=dim)
        dict_elements_2 = torch.nn.functional.normalize(dict_elements_2, p=p, dim=dim)

    # Compute cosine similarity using matrix multiplication
    cosine_sim: jaxtyping.Float[torch.Tensor, "d_llm d_llm"] = torch.mm(
        dict_elements_1, dict_elements_2.T
    )
    # max_cosine_sim, _ = torch.max(cosine_sim, dim=1)
    return cosine_sim


def obtain_cos_sim(
    sae_list: torch.nn.Module, model: sae_lens.HookedSAETransformer = None
):
    if model:
        cos_sim = torch.zeros(
            len(sae_list), sae_list[0].cfg.d_sae, sae_list[0].cfg.d_sae
        )
        for layer in range(len(sae_list)):
            cos_sim[layer] = get_cosine_similarity(
                sae_list[layer].W_dec, sae_list[layer].W_dec
            )
    else:
        unembedding_matrix = model.unembed.W_U
        cos_sim = torch.zeros(
            len(sae_list), sae_list[0].cfg.d_sae, unembedding_matrix.shape[1]
        )
        for layer in range(len(sae_list)):
            cos_sim[layer] = get_cosine_similarity(
                sae_list[layer].W_dec, unembedding_matrix.T, normalized=False
            )

    return cos_sim


def plot_freq(activation: torch.Tensor):
    fig = px.box(
        activation.T.cpu().numpy(),
        title="Frequency of activations in the model",
        labels={"value": "Frequency", "variable": "Layer"},
    )
    fig.write_html("./res/freq_box.html")

def plot_cos_sim(cos_sim: torch.Tensor):
    min_cos_sim = cos_sim.fill_diagonal_(100).min(dim=1).values.cpu().numpy()
    max_cos_sim = cos_sim.fill_diagonal_(-100).max(dim=1).values.cpu().numpy()
    min_fig = px.box(min_cos_sim, title="Min cosine similarity", labels={"value": "Cosine similarity", "variable": "Layer"})
    min_fig.write_html("./res/min_cos_sim_box.html")
    max_fig = px.box(max_cos_sim, title="Max cosine similarity", labels={"value": "Cosine similarity", "variable": "Layer"})
    max_fig.write_html("./res/max_cos_sim_box.html")
    return min_cos_sim, max_cos_sim

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger()
    device = get_device()
    logger.info(f"Using device {device}")
    logger.info(f"all arguments: {args}")
    logger.info(f"step 1: Load model, data and sae to {device}")

    sae_list, model, dataset = load_sae_from_saelens(args.sae_name, device)
    logger.info(f"loaded {len(sae_list)} saes from {args.sae_name}")
    logger.info(f"loaded model {model}")
    logger.info(f"loaded dataset {dataset}")

    logger.info(f"step 2: get the activation of the SAE")
    activations = obtain_activations(
        sae_list, model, dataset, activation_path=args.activation_path
    )
    logger.info(f"obtained activations of shape {activations.shape}")
    logger.info(f"step 3: Geometry analysis")
    logger.info(f"step 3.1: vectors' cos sim in the same layer(max and min)")
    cos_sim = obtain_cos_sim(sae_list)
    _, _ = plot_cos_sim(cos_sim)
    # TODO: pairwise in the same layer, too large to plot
    logger.info(f"step 3.2: cos sim with unembedding matrix")
    cos_sim_unembedding = obtain_cos_sim(sae_list, model)
    _, _ = plot_cos_sim(cos_sim_unembedding)
    # TODO: here we do not care about the meaning, we only care about the cos sim and freq
    logger.info(f"step 4: Frequency analysis")
    logger.info(f"step 4.1: Plot avg frequency of the activation of the SAE")
    plot_freq(activation=activations)
    logger.info(f"step 4.2: cos sim with high freq, low freq and between them")
    logger.info(f"step 4.3: freq of the high cos sim, low cos sim")
    logger.info(
        f"step 4.4: freq of the high cos sim, low cos sim between the unembedding matrix"
    )

    logger.info(f"Then we can save the results and see the ablation study")
    logger.info(f"step 5: use different kinds of dataset to see the difference")
