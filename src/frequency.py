import torch
import argparse
import sae_lens
import transformer_lens
import datasets
import os
import jaxtyping
import sklearn
import pandas as pd
import plotly.express as px
import plotly.colors as pc
from tqdm import tqdm
from typing import Tuple, List
from logging import Logger
from utils import set_seed, get_device
from logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed value")
    parser.add_argument(
        "--sae_name",
        type=str,
        default="pythia-70m-deduped-res-sm",
        help="SAE model name",
        choices=[
            "pythia-70m-deduped-res-sm",
            "gpt2-small-res",
            "get2-medium-res",
            "gemma-scope-2b-pt-res",
            "pythia-70m-deduped-att-sm",
            "pythia-70m-deduped-mlp-sm",
        ],
    )
    parser.add_argument(
        "--activation_path",
        type=str,
        default="./res/tmp.pt",
        help="Path to save the activation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="abstract_math",
        choices=["wikitext", "abstract_math"],
        help="Path to save the frequency",
    )
    parser.add_argument(
        "--use_error_term",
        type=bool,
        default=True,
        help="Whether to use the error term in the SAE",
    )
    return parser.parse_args()


@torch.no_grad()
def load_sae_from_saelens(
    sae_name: str, device: str = "cuda", dataset_name: str = "wikitext"
) -> Tuple[torch.nn.Module, sae_lens.HookedSAETransformer, torch.utils.data.Dataset]:
    sae_list = []
    match sae_name:
        case "pythia-70m-deduped-res-sm":
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
        case "pythia-70m-deduped-att-sm":
            for layer in range(layers):
                layers = 6
                release = "pythia-70m-deduped-att-sm"
                model_name = "pythia-70m-deduped"
                for layer in range(layers):
                    sae_id = f"blocks.{layer}.hook_hook_attn_out"
                    sae_list.append(
                        sae_lens.SAE.from_pretrained(
                            release=release, sae_id=sae_id, device=device
                        )[0]
                    )
            model = sae_lens.HookedSAETransformer.from_pretrained(model_name).to(device)
        case "pythia-70m-deduped-mlp-sm":
            pass
        case "gpt2-small-res":
            pass
        case "get2-medium-res":
            pass
        case "gemma-scope-2b-pt-res":
            layers = 26
            for layer in range(layers):
                release = "gemma-scope-2b-pt-res-canonical"
                sae_id = f"layer_{layer}/width_16k/canonical"
                sae_list.append(
                    sae_lens.SAE.from_pretrained(
                        release=release, sae_id=sae_id, device=device
                    )[0]
                )
            model = sae_lens.HookedSAETransformer.from_pretrained("gemma-2-2b").to(
                device
            )
        case "llama":
            pass
        case _:
            raise ValueError(f"Invalid SAE model name: {sae_name}")
    match dataset_name:
        case "wikitext":
            dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
                "train"
            ]
        case "abstract_math":
            dataset = datasets.load_dataset("hbin0701/abstract_math")["train"]["input"]
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
    return sae_list, model, dataset


def obtain_activations(
    sae_list: torch.nn.Module,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    save_name: str = "tmp.pt",
    data_name: str = "abstract_math",
    args: argparse.Namespace = None,
) -> torch.Tensor:
    "output: (num_layers, num_features)"
    if save_name and os.path.exists(save_name):
        return torch.load(save_name, weights_only=True)
    match sae_list[0].cfg.model_name:
        case "pythia-70m-deduped":
            doc_len = 0
            freq_mean_global = 0
            layers = 6
            freqs = torch.zeros(layers + 1, sae_list[0].cfg.d_sae).to(device)
            if data_name == "wikitext":
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
                            freq[layer] = (cache[prompt] > 1e-3)[0].sum(
                                0
                            ) / local_doc_len
                        freq[layer + 1] = (cache[prompt2] > 1e-3)[0].sum(0) / local_doc_len
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
            elif data_name == "abstract_math":
                doc_len = int(len(dataset) * 0.01)
                for idx in tqdm(range(doc_len)):
                    example = dataset[idx]
                    tokens = model.to_tokens([example], prepend_bos=True)
                    if args.use_error_term:
                        _, cache = model.run_with_cache_with_saes(
                            tokens, saes=sae_list, use_error_term=True
                        )
                    else:
                        _, cache = model.run_with_cache_with_saes(tokens, saes=sae_list)
                    local_doc_len = cache[
                        "blocks.0.hook_resid_post.hook_sae_acts_post"
                    ].shape[1]
                    freq = torch.zeros_like(freqs)
                    for layer in range(layers):
                        prompt = f"blocks.{layer}.hook_resid_pre.hook_sae_acts_post"
                        prompt2 = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
                        if layer == 0:
                            freq[layer] = (cache[prompt] > 1e-3)[0].sum(
                                0
                            ) / local_doc_len
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
        case "gpt2":
            pass
        case "gemma-2-2b":
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
                    prompt2 = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
                    freq[layer + 1] = (cache[prompt2] > 1e-3)[0].sum(0) / local_doc_len
                new_doc_len = doc_len + local_doc_len
                if idx == 0:
                    freq_mean_global = freq
                else:
                    freq_mean_global = (
                        freq_mean_global * doc_len / new_doc_len
                        + freq * local_doc_len / new_doc_len
                    )
                doc_len = new_doc_len
        case _:
            raise ValueError(f"Invalid model name: {sae_list[0].cfg.model_name}")

    if os.path.exists(save_name):
        os.remove(save_name)
    torch.save(freq_mean_global, save_name)
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


def plot_freq(activation: torch.Tensor, model_name: str = None, data_name: str = None):
    fig = px.box(
        activation.T.cpu().numpy(),
        title="Frequency of activations with extreme low frequency in the model",
        labels={"value": "Frequency", "variable": "Layer"},
    )
    fig.write_html(f"./res/{data_name}_{model_name}_freq_w_zero.html")


def plot_cos_sim(
    sae_list: List[sae_lens.SAE],
    is_umbedding: bool = False,
    model: sae_lens.HookedSAETransformer = None,
    model_name: str = None,
    by_layer: bool = False,
):
    model_name = model_name
    colors = pc.n_colors("rgb(5, 200, 200)", "rgb(200, 10, 10)", 13, colortype="rgb")
    min_cos_sim_stats = []
    max_cos_sim_stats = []
    for layer in range(len(sae_list)):
        if is_umbedding:
            cos_sim = get_cosine_similarity(sae_list[layer].W_dec, model.unembed.W_U.T)
        elif by_layer:
            cos_sim = get_cosine_similarity(
                sae_list[layer].W_dec, sae_list[layer + 1].W_dec
            )
        else:
            cos_sim = get_cosine_similarity(
                sae_list[layer].W_dec, sae_list[layer].W_dec
            )

        min_df = pd.DataFrame(
            {
                "cos": cos_sim.fill_diagonal_(100).min(dim=1).values.cpu().numpy(),
                "layer": layer,
            }
        )
        max_df = pd.DataFrame(
            {
                "cos": cos_sim.fill_diagonal_(-100).max(dim=1).values.cpu().numpy(),
                "layer": layer,
            }
        )
        min_cos_sim_stats.append(min_df)
        max_cos_sim_stats.append(max_df)
        if by_layer and layer == len(sae_list) - 2:
            break
    min_cos_sim = pd.concat(min_cos_sim_stats, axis=0)
    max_cos_sim = pd.concat(max_cos_sim_stats, axis=0)
    if is_umbedding:
        max_title = (
            "Max cosine similarity between the unembedding matrix and decoder weights"
        )
        min_title = (
            "Min cosine similarity between the unembedding matrix and decoder weights"
        )
        max_output = f"./res/{model_name}_max_cos_sim_unembedding_box.html"
        min_output = f"./res/{model_name}_min_cos_sim_unembedding_box.html"
    elif by_layer:
        max_title = (
            "Max cosine similarity between the decoder weights in different layers"
        )
        min_title = (
            "Min cosine similarity between the decoder weights in different layers"
        )
        max_output = f"./res/{model_name}_max_cos_sim_by_layer_box.html"
        min_output = f"./res/{model_name}_min_cos_sim_by_layer_box.html"
    else:
        max_title = "Max cosine similarity between the decoder weights"
        min_title = "Min cosine similarity between the decoder weights"
        max_output = f"./res/{model_name}_max_cos_sim_box.html"
        min_output = f"./res/{model_name}_min_cos_sim_box.html"
    max_fig = px.box(
        max_cos_sim,
        x="layer",
        y="cos",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title=max_title,
        labels={"cos": "cos", "layer": "Layer"},
    )
    min_fig = px.box(
        min_cos_sim,
        x="layer",
        y="cos",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title=min_title,
        labels={"cos": "cos", "layer": "Layer"},
    )
    min_fig.write_html(min_output)
    max_fig.write_html(max_output)
    return min_cos_sim, max_cos_sim


def plot_freq2cos(
    freqs: torch.Tensor,
    sae_list: List[sae_lens.SAE],
    logger: Logger,
    model_name: str = None,
    is_umbedding: bool = False,
    model: sae_lens.HookedSAETransformer = None,
    dataset_name: str = None,
):
    top_stats = []
    bottom_stats = []
    inter_stats = []
    top_min_stats = []
    bottom_min_stats = []
    inter_min_stats = []
    colors = pc.n_colors("rgb(5, 200, 200)", "rgb(200, 10, 10)", 13, colortype="rgb")
    for layer in range(len(sae_list)):
        freq = freqs[layer]
        if dataset_name == "wikitext":
            top_1_percent_values, top_1_percent_indices = torch.topk(
                freq, int(0.1 * freq.numel()), largest=True
            )
            bottom_1_percent_values, bottom_1_percent_indices = torch.topk(
                freq, int(0.1 * freq.numel()), largest=False
            )
        else:
            top_1_percent_values, top_1_percent_indices = torch.topk(
                freq, int(0.01 * freq.numel()), largest=True
            )
            bottom_1_percent_values, bottom_1_percent_indices = torch.topk(
                freq, int(0.01 * freq.numel()), largest=False
            )

        logger.info(
            f"layer {layer} top 1% freq indices from {top_1_percent_values.max()} to {top_1_percent_values.min()}"
        )
        logger.info(
            f"layer {layer} bottom 1% freq indices from {bottom_1_percent_values.max()} to {bottom_1_percent_values.min()}"
        )
        if is_umbedding:
            top_cos_sim = get_cosine_similarity(
                sae_list[layer].W_dec[top_1_percent_indices],
                model.unembed.W_U.T,
            )
            bottom_cos_sim = get_cosine_similarity(
                sae_list[layer].W_dec[bottom_1_percent_indices],
                model.unembed.W_U.T,
            )
            inter_cos_sim = get_cosine_similarity(
                sae_list[layer].W_dec[top_1_percent_indices],
                model.unembed.W_U.T,
            )
        else:
            top_cos_sim = get_cosine_similarity(
                sae_list[layer].W_dec[top_1_percent_indices],
                sae_list[layer].W_dec[top_1_percent_indices],
            )
            bottom_cos_sim = get_cosine_similarity(
                sae_list[layer].W_dec[bottom_1_percent_indices],
                sae_list[layer].W_dec[bottom_1_percent_indices],
            )
            inter_cos_sim = get_cosine_similarity(
                sae_list[layer].W_dec[top_1_percent_indices],
                sae_list[layer].W_dec[bottom_1_percent_indices],
            )
        top_stats.append(
            pd.DataFrame(
                {
                    "cos": top_cos_sim.fill_diagonal_(-100)
                    .max(dim=1)
                    .values.cpu()
                    .numpy(),
                    "layer": layer,
                }
            )
        )
        top_min_stats.append(
            pd.DataFrame(
                {
                    "cos": top_cos_sim.fill_diagonal_(100)
                    .min(dim=1)
                    .values.cpu()
                    .numpy(),
                    "layer": layer,
                }
            )
        )
        bottom_stats.append(
            pd.DataFrame(
                {
                    "cos": bottom_cos_sim.fill_diagonal_(-100)
                    .max(dim=1)
                    .values.cpu()
                    .numpy(),
                    "layer": layer,
                }
            )
        )
        bottom_min_stats.append(
            pd.DataFrame(
                {
                    "cos": bottom_cos_sim.fill_diagonal_(100)
                    .min(dim=1)
                    .values.cpu()
                    .numpy(),
                    "layer": layer,
                }
            )
        )
        inter_stats.append(
            pd.DataFrame(
                {
                    "cos": inter_cos_sim.fill_diagonal_(-100)
                    .max(dim=1)
                    .values.cpu()
                    .numpy(),
                    "layer": layer,
                }
            )
        )
        inter_min_stats.append(
            pd.DataFrame(
                {
                    "cos": inter_cos_sim.fill_diagonal_(100)
                    .min(dim=1)
                    .values.cpu()
                    .numpy(),
                    "layer": layer,
                }
            )
        )
    top_df = pd.concat(top_stats, axis=0)
    bottom_df = pd.concat(bottom_stats, axis=0)
    inter_df = pd.concat(inter_stats, axis=0)
    top_min_df = pd.concat(top_min_stats, axis=0)
    bottom_min_df = pd.concat(bottom_min_stats, axis=0)
    inter_min_df = pd.concat(inter_min_stats, axis=0)
    top_fig = px.box(
        top_df,
        x="layer",
        y="cos",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title="Top 1% freq max cosine similarity",
        labels={"cos": "cos", "layer": "Layer"},
    )
    bottom_fig = px.box(
        bottom_df,
        x="layer",
        y="cos",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title="Bottom 1% freq max cosine similarity",
        labels={"cos": "cos", "layer": "Layer"},
    )
    inter_fig = px.box(
        inter_df,
        x="layer",
        y="cos",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title="max cosine similarity from the top 1% freq to the bottom 1% freq",
        labels={"cos": "cos", "layer": "Layer"},
    )
    top_min_fig = px.box(
        top_min_df,
        x="layer",
        y="cos",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title="Top 1% freq min cosine similarity",
        labels={"cos": "cos", "layer": "Layer"},
    )
    bottom_min_fig = px.box(
        bottom_min_df,
        x="layer",
        y="cos",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title="Bottom 1% freq min cosine similarity",
        labels={"cos": "cos", "layer": "Layer"},
    )
    inter_min_fig = px.box(
        inter_min_df,
        x="layer",
        y="cos",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title="min cosine similarity from the top 1% freq to the bottom 1% freq",
        labels={"cos": "cos", "layer": "Layer"},
    )
    if is_umbedding:
        top_min_fig.write_html(
            f"./res/{dataset_name}_{model_name}_top_min_unembedding_cos_sim_box.html"
        )
        bottom_min_fig.write_html(
            f"./res/{dataset_name}_{model_name}_bottom_min_unembedding_cos_sim_box.html"
        )
        # inter_min_fig.write_html(f"./res/{model_name}_inter_min_unembedding_cos_sim_box.html")
        top_fig.write_html(
            f"./res/{dataset_name}_{model_name}_top_unembedding_cos_sim_box.html"
        )
        bottom_fig.write_html(
            f"./res/{dataset_name}_{model_name}_bottom_unembedding_cos_sim_box.html"
        )
        # inter_fig.write_html(f"./res/{model_name}_inter_unembedding_cos_sim_box.html")
    else:
        top_min_fig.write_html(
            f"./res/{dataset_name}_{model_name}_top_min_cos_sim_box.html"
        )
        bottom_min_fig.write_html(
            f"./res/{dataset_name}_{model_name}_bottom_min_cos_sim_box.html"
        )
        inter_min_fig.write_html(
            f"./res/{dataset_name}_{model_name}_inter_min_cos_sim_box.html"
        )
        top_fig.write_html(f"./res/{dataset_name}_{model_name}_top_cos_sim_box.html")
        bottom_fig.write_html(
            f"./res/{dataset_name}_{model_name}_bottom_cos_sim_box.html"
        )
        inter_fig.write_html(
            f"./res/{dataset_name}_{model_name}_inter_cos_sim_box.html"
        )
    return top_df, bottom_df, inter_df


def plot_cos2freq(
    activations: torch.Tensor, sae_list: List[sae_lens.SAE], logger: Logger
):
    top_stats = []
    bottom_stats = []
    inter_stats = []
    colors = pc.n_colors("rgb(5, 200, 200)", "rgb(200, 10, 10)", 13, colortype="rgb")
    for layer in range(len(sae_list)):
        cos_sim = get_cosine_similarity(sae_list[layer].W_dec, sae_list[layer].W_dec)
        top_1_percent_values, top_1_percent_indices = cos_sim.fill_diagonal_(-100).topk(
            k=int(0.01 * cos_sim.shape[1]), dim=1, largest=True
        )
        bottom_1_percent_values, bottom_1_percent_indices = cos_sim.fill_diagonal_(
            100
        ).topk(k=int(0.01 * cos_sim.shape[1]), dim=1, largest=False)
        logger.info(
            f"layer {layer} top 1% cos sim indices from {top_1_percent_values.max()} to {top_1_percent_values.min()}"
        )
        logger.info(
            f"layer {layer} bottom 1% cos sim indices from {bottom_1_percent_values.max()} to {bottom_1_percent_values.min()}"
        )
        top_1_percent_indices = torch.unique(top_1_percent_indices)
        bottom_1_percent_indices = torch.unique(bottom_1_percent_indices)
        top_stats.append(
            pd.DataFrame(
                {
                    "freq": activations[layer][top_1_percent_indices].cpu().numpy(),
                    "layer": layer,
                }
            )
        )
        bottom_stats.append(
            pd.DataFrame(
                {
                    "freq": activations[layer][bottom_1_percent_indices].cpu().numpy(),
                    "layer": layer,
                }
            )
        )
    top_df = pd.concat(top_stats, axis=0)
    bottom_df = pd.concat(bottom_stats, axis=0)
    top_fig = px.box(
        top_df,
        x="layer",
        y="freq",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title="Top 1% cos sim's freq",
        labels={"freq": "freq", "layer": "Layer"},
    )
    bottom_fig = px.box(
        bottom_df,
        x="layer",
        y="freq",
        width=800,
        height=600,
        color="layer",
        color_discrete_sequence=colors,
        title="Bottom 1% cos sim's freq",
        labels={"freq": "freq", "layer": "Layer"},
    )
    top_fig.write_html("./res/top_cos2freq_box.html")
    bottom_fig.write_html("./res/bottom_cos2freq_box.html")
    return top_df, bottom_df


@torch.no_grad()
def plot_w_pca(
    sae_list: List[sae_lens.SAE],
    acts: torch.Tensor,
    sae_name: str,
    plot_extreme: bool = False,
    percent: float = 1e-4,
    largest: bool = True,
):
    # TODO: a trivil draw, no need to plot because pca is not useful for the decoder weights
    if largest:
        _, extreme_freq_indices = torch.topk(
            acts, int(percent * acts.numel()//7), largest=True
        )
    else:
        _, extreme_freq_indices = torch.topk(
            acts, int(percent * acts.numel()//7), largest=False
        )       

    pca = sklearn.decomposition.PCA(n_components=256)

    colors = pc.n_colors("rgb(5, 200, 200)", "rgb(200, 10, 10)", 6, colortype="rgb")
    labels = {str(i): f"PC {i+1}" for i in range(2)}
    labels["color"] = "Median Price"
    pca_stat = []
    for layer in range(6):
        pca_res = pca.fit_transform(sae_list[layer].W_dec[extreme_freq_indices[layer], :].cpu().detach().numpy())
        pca_stat.append(
            pd.DataFrame({"layer": layer, "pc1": pca_res[:, 0], "pc2": pca_res[:, 1]})
        )
    pca_stat = pd.concat(pca_stat, axis=0)
    colors = pc.n_colors("rgb(5, 200, 200)", "rgb(200, 10, 10)", 6, colortype="rgb")
    labels = {str(i): f"PC {i+1}" for i in range(2)}
    total_var = pca.explained_variance_ratio_.sum() * 100
    fig = px.scatter(
        pca_stat,
        x="pc1",
        y="pc2",
        color="layer",
        labels=labels,
        color_discrete_sequence=colors,
        title=f"Total Explained Variance: {total_var:.2f}%",
    )
    fig.show()


def plot_freq_diff(activation_path):
    error = "./res/reconstuction_math_pythia_70m_deduped_res_sm_freq_mean_global.pt"
    non_error = "./res/abstract_math_pythia-70m-deduped-res-sm_abl_freq_mean_global_extreme.pt"
    activations = torch.load(non_error, weights_only=True)
    err_activations = torch.load(error, weights_only=True)
    freq_diff = activations - err_activations
    fig = px.box(
        freq_diff.T.cpu().numpy(),
        title="Frequency difference between the error term and the non-error term",
        labels={"value": "Frequency", "variable": "Layer"},
    )
    fig.write_html(f"./res/tmp_freq_diff_math_abl_3_low_freq.html")

@torch.no_grad()    
def ablation_extreme_freq(
    activation: torch.Tensor,
    model_name: str,
    data_name: str,
    dataset: torch.utils.data.Dataset,
    sae_list: List[sae_lens.SAE],
    largest: bool = True,
    percent: float = 1,
):
    if largest:
        _, extreme_freq_indices = torch.topk(
            activation, int(percent * activation.numel()//7), largest=True
        )
        print(extreme_freq_indices.shape)
    else:
        _, extreme_freq_indices = torch.topk(
            activation, int(percent * activation.numel()//7), largest=False
        )
    for idx in range(len(sae_list)):
        print(sae_list[idx].W_dec.nonzero().shape)
        sae_list[idx].W_dec.requires_grad = False
        list(map(lambda idy: sae_list[idx].W_dec[idy, :].zero_(), extreme_freq_indices[0]))
    doc_len = 0
    freq_mean_global = 0
    layers = 6
    freqs = torch.zeros(layers + 1, sae_list[0].cfg.d_sae).to(device)
    doc_len = int(len(dataset) * 0.01)
    idx = 5
    print(sae_list[idx].W_dec.nonzero().shape)

    for idx in tqdm(range(doc_len)):
        example = dataset[idx]
        tokens = model.to_tokens([example], prepend_bos=True)
        _, cache = model.run_with_cache_with_saes(
            tokens, saes=sae_list, use_error_term=True
        )
        local_doc_len = cache["blocks.0.hook_resid_post.hook_sae_acts_post"].shape[
            1
        ]
        freq = torch.zeros_like(freqs)
        for layer in range(layers):
            prompt = f"blocks.{layer}.hook_resid_pre.hook_sae_acts_post"
            prompt2 = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
            if layer == 0:
                freq[layer] = (cache[prompt] > 1e-3)[0].sum(0) / local_doc_len
            else:
                freq[layer + 1] = (cache[prompt2] > 1e-3)[0].sum(0) / local_doc_len
        new_doc_len = doc_len + local_doc_len
        if idx == 0:
            freq_mean_global = freq
        else:
            freq_mean_global = (
                freq_mean_global * doc_len / new_doc_len
                + freq * local_doc_len / new_doc_len
            )
        doc_len = new_doc_len
    torch.save(
        freq_mean_global,
        f"./res/{data_name}_{model_name}_abl_freq_mean_global_extreme.pt",
    )
    return freq_mean_global


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger()
    device = get_device()
    logger.info("-" * 60)
    logger.info(f"Start running frequency analysis")
    logger.info(f"Using device {device}")
    logger.info(f"all arguments: {args}")
    logger.info(f"step 1: Load model, data and sae to {device}")

    sae_list, model, dataset = load_sae_from_saelens(
        args.sae_name, device, args.dataset
    )
    logger.info(f"loaded {len(sae_list)} saes from {args.sae_name}")
    logger.info(f"loaded model {model}")
    logger.info(f"loaded dataset {args.dataset}")

    logger.info(f"step 2: get the activation of the SAE")
    activations = obtain_activations(
        sae_list,
        model,
        dataset,
        save_name=args.activation_path,
        data_name=args.dataset,
        args=args,
    )
    logger.info(f"obtained activations of shape {activations.shape}")
    logger.info(f"step 3: Geometry analysis")
    # logger.info(f"step 3.1: vectors' cos sim in the same layer(max and min)")

    # cos_sim = obtain_cos_sim(sae_list)
    # _, _ = plot_cos_sim(sae_list, is_umbedding=False, model_name=args.sae_name)
    # TODO: pairwise in the same layer, too large to plot
    # logger.info(f"step 3.2: cos sim with unembedding matrix")

    # _, _ = plot_cos_sim(
    #    sae_list=sae_list, model=model, is_umbedding=True, model_name=args.sae_name
    # )
    # logger.info(f"step 3.3: cos sim by layer")
    # _, _ = plot_cos_sim(
    #    sae_list, is_umbedding=False, model_name=args.sae_name, by_layer=True
    # )
    # TODO: here we do not care about the meaning, we only care about the cos sim and freq
    # logger.info(f"step 4: Frequency analysis")
    logger.info(f"step 4.1: Plot avg frequency of the activation of the SAE")
    # plot_freq(activation=activations, savemodel_name=args.sae_name, data_name=args.dataset, err=args.use_error_term)
    # logger.info(f"step 4.2: cos sim with high freq, low freq and between them")
    # plot_freq2cos(activations, sae_list, logger=logger, model_name=args.sae_name, dataset_name=args.dataset)
    # logger.info(f"step 4.3: freq of the high cos sim, low cos sim")
    # plot_cos2freq(activations, sae_list, logger=logger)
    # logger.info(f"step 4.4: freq of the high cos sim, low cos sim between the unembedding matrix")
    # plot_freq2cos(activations, sae_list, logger=logger, model_name=args.sae_name, is_umbedding=True, model=model)
    # logger.info(f"Then we can save the results and see the ablation study")
    logger.info(f"step 5: ablation study")
    #logger.info(
    #    f"step 5.1: freq difference between the error term and the non-error term"
    #)
    #plot_freq_diff(activation_path=args.activation_path)
    logger.info(f"step 5.2: ablation of the extreme freq")
    ablation_extreme_freq(
        activation=activations,
        model_name=args.sae_name,
        data_name=args.dataset,
        dataset=dataset,
        sae_list=sae_list,
        largest=True,
    )
    plot_freq_diff(activation_path=args.activation_path)
    # logger.info(f"step 5: PCA of the decoder weights")
    # plot_w_pca(sae_list, activations, sae_name=args.sae_name, plot_extreme=True)
    logger.info("end of the frequency analysis")
    logger.info("-" * 60)
