import torch
import jaxtyping
import pandas as pd
import seaborn as sns
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from data import load_dataset_by_name, fetch_sae_and_model

torch.set_grad_enabled(False)


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


def cs_freqPattern_comparsion(model_name: str, dataset_name: str):
    """
    1. load saes, models
    2. compute the cs and select the top cos sim pair and min cos sim pair
    3. compare the given freq pattern in the top and min cos sim pair
    4. draw a box plot
    """

    saes, model = fetch_sae_and_model(model_name)
    dataset, ratio, text = load_dataset_by_name(dataset_name)
    max_diffs = [[] for _ in range(len(saes))]
    min_diffs = [[] for _ in range(len(saes))]
    zero_diffs = [[] for _ in range(len(saes))]
    dataset_length = int(len(dataset) * ratio)
    max_indices = [[[], []] for _ in range(len(saes))]

    for idx in tqdm(range(dataset_length)):
        example = dataset[idx]
        tokens = model.to_tokens([example[text]], prepend_bos=True)
        loss1, cache1 = model.run_with_cache_with_saes(tokens, saes=saes)
        model.reset_saes()
        for layer in range(len(saes)):
            cs = get_cosine_similarity(saes[layer].W_dec, saes[layer].W_dec).fill_diagonal_(-100).cpu()
            # max: 0.6
            # min: -0.7
            # zero: 0.001
            max_indice = torch.where((cs > -0.1) & (cs < 0.1))
            max_indices[layer][0].extend(max_indice[0][:300])
            max_indices[layer][1].extend(max_indice[1][:300])
            dim1, dim2 = len(max_indices[layer][0]), len(max_indices[layer][1])
            max_cs_pair_num = min(dim1, dim2)
            if model_name == "pythia-70m-deduped" or model_name == "gemma-2-2b":
                prompt = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
            for idy in range(max_cs_pair_num):
                res1 = cache1[prompt][:, :, max_indices[layer][0][idy]] > 1e-4
                res2 = cache1[prompt][:, :, max_indices[layer][1][idy]] > 1e-4
                if res1.sum() == 0 and res2.sum() == 0:
                    pass
                else:
                    max_diffs[layer].append(
                        ((res1 & res2).sum() / (res1 | res2).sum()).cpu().item()
                    )
    stats = {"layer": [], "max": [], "avg": [], "min": [], "25%": [], "75%": []}

    for layer in range(26):
        stats["layer"].append(layer)
        stats["max"].append(max(max_diffs[layer]))
        stats["avg"].append(sum(max_diffs[layer]) / len(max_diffs[layer]))
        stats["min"].append(min(max_diffs[layer]))
        stats["25%"].append(np.percentile(max_diffs[layer], 25))
        stats["75%"].append(np.percentile(max_diffs[layer], 75))

    df = pd.DataFrame(stats)
    fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="layer", y="max", label="max", ax=ax)
    sns.lineplot(data=df, x="layer", y="avg", label="avg", ax=ax)
    # sns.lineplot(data=df, x="layer", y="min", label="min", ax=ax)
    sns.lineplot(data=df, x="layer", y="25%", label="25%", ax=ax)
    sns.lineplot(data=df, x="layer", y="75%", label="75%", ax=ax)
    ax.set_title(f"Activation Statistics for {model_name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Activation ratio")
    plt.savefig(f"{model_name}_high_cs_activation_patterns.pdf")
    
    # plot encoder
    cs = [[] for _ in range(26)]
    for layer in range(len(saes)):
        dim1, dim2 = len(max_indices[layer][0]), len(max_indices[layer][1])
        max_cs_pair_num = min(dim1, dim2)
        p = 2
        dim = 1
        normalized_enc = torch.nn.functional.normalize(saes[layer].W_enc, p=p, dim=dim)
        for idy in range(max_cs_pair_num):
            cosine_sim = torch.mm(normalized_enc[:, max_indices[layer][0][idy]].unsqueeze(0), normalized_enc[:, max_indices[layer][1][idy]].unsqueeze(1))
            cs[layer].append(cosine_sim.cpu().item())
    cs_df = pd.DataFrame({
        "layer": np.repeat(range(26), [len(layer_cs) for layer_cs in cs]),
        "cosine similarity": [item for sublist in cs for item in sublist]
    })

    sns.boxplot(data=cs_df, x="layer", y="cosine similarity")
    plt.title("Cosine Similarity of Encoder Weights by Layer")
    plt.show()
    plt.savefig("zero_cosine_similarity_boxplot.pdf")
    return None


def name2lrc(name: str) -> Tuple[int, int, int]:
    if name == "gemma-2-2b":
        return 26, 7, 4
    elif name == "llama3.1-8b":
        32, 8, 4
    elif name == "pythia-70m-deduped":
        return 6, 2, 3


def plot_freq(
    model_name: str = "gemma-2-2b",  # "llama3.1-8b", "pythia-70m-deduped", "gemma-2-2b"
) -> None:
    """
    Plot the frequency of activations.
    Example:
    plot_freq("gemma-2-2b")
    """
    layers, row, col = name2lrc(model_name)
    acts = load_acts_from_pretrained()
    stats = {"layer": [], "max": [], "avg": [], "min": [], "25%": [], "75%": []}
    avg_acts = (acts[0] + acts[1] + acts[2]) / 3
    for layer in range(layers):
        stats["layer"].append(layer)
        stats["max"].append(avg_acts[layer].max().item())
        stats["avg"].append(avg_acts[layer].mean().item())
        stats["min"].append(avg_acts[layer].min().item())
        stats["25%"].append(avg_acts[layer].quantile(0.25).item())
        stats["75%"].append(avg_acts[layer].quantile(0.75).item())
    df = pd.DataFrame(stats)
    fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="layer", y="max", label="max", ax=ax)
    sns.lineplot(data=df, x="layer", y="avg", label="avg", ax=ax)
    # sns.lineplot(data=df, x="layer", y="min", label="min", ax=ax)
    sns.lineplot(data=df, x="layer", y="25%", label="25%", ax=ax)
    sns.lineplot(data=df, x="layer", y="75%", label="75%", ax=ax)
    ax.set_title(f"Activation Statistics for {model_name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Activation ratio")
    plt.savefig(f"../res/freq/{model_name}_activation_statistics.pdf")
    plt.show()
    return None


def load_acts_from_pretrained(
    model_name: str = "gemma2",
    path: str = "./res/acts/",
    data_name: List = ["math", "code", "wiki"],
) -> List[torch.Tensor]:
    """
    Load activations from pretrained model.
    """
    data_name = ["math", "code", "wiki"]
    acts = []
    for data in data_name:
        full_path = osp.join(path, f"{model_name}_freqs_{data}.pt")
        acts.append(torch.load(full_path))
    return acts


if __name__ == "__main__":
    max_diffs, min_diffs, zero_diffs = cs_freqPattern_comparsion(
        "pythia-70m-deduped", "MMLU"
    )
