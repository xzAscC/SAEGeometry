import sae_lens
import torch
import datasets
import argparse
import random
import jaxtyping
import warnings
import copy
import pandas as pd
import numpy as np
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
from torch import autocast
from typing import List, Tuple
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


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
        default= "llama3",
        help="The model name.",
        choices=["gemma2", "llama3", "pythia"],
    )
    parser.add_argument(
        "--use_error_term",
        type=bool,
        default=False,
        help="Whether to use the error term.",
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
    ds = ["math"]
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
        acts.append(torch.load(f"{path}{model_name}_freqs_{data}.pt"))
    return acts


def plot_freq(
    acts: torch.Tensor,
    model_name: str = "gemma2",
    data_name: List = ["math", "code", "wiki"],
) -> None:
    """
    Plot the frequency of activations.
    """
    layers, row, col = name2lrc(model_name)
    fig, axes = plt.subplots(row, col, figsize=(col * 10, row * 10))
    for layer in range(layers):
        ax = axes[layer // col, layer % col]
        for idx in range(len(data_name)):
            data = data_name[idx]
            df = pd.DataFrame(
                {
                    "vector index": range(len(acts[idx][layer])),
                    "frequency value": acts[idx][layer].cpu().numpy(),
                }
            )
            sns.lineplot(
                data=df,
                ax=ax,
                x="vector index",
                y="frequency value",
                label=f"{data} Freq",
            )
        ax.set_title(f"Layer {layer}")
    # plt.show()
    if model_name == "gemma2":
        fig.delaxes(axes[6, 3])
        fig.delaxes(axes[6, 2])
    fig.savefig(f"./res/freq/{model_name}_freqs.pdf")
    plt.close(fig)
    return None


def name2lrc(name: str) -> Tuple[int, int, int]:
    if name == "gemma2":
        return 26, 7, 4
    elif name == "llama3":
        return 32, 8, 4
    elif name == "pythia":
        return 6, 2, 3


def plot_cs(
    saes: List[sae_lens.SAE],
    model_name: str = "gemma2",
) -> None:
    cs_min_stats = []
    cs_max_stats = []
    layers, _, _ = name2lrc(model_name)
    for layer in range(layers):
        cs_max_stats.append(
            pd.DataFrame(
                {
                    "cos": get_cosine_similarity(saes[layer].W_dec, saes[layer].W_dec)
                    .fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                }
            )
        )
        cs_min_stats.append(
            pd.DataFrame(
                {
                    "cos": get_cosine_similarity(saes[layer].W_dec, saes[layer].W_dec)
                    .fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                }
            )
        )
    max_fig = sns.boxplot(pd.concat(cs_max_stats, axis=0), x="layer", y="cos")
    min_fig = sns.boxplot(pd.concat(cs_min_stats, axis=0), x="layer", y="cos")
    min_fig.get_figure().savefig(f"./res/cos_sim/{model_name}_cs.pdf")
    plt.close(max_fig.get_figure())
    plt.close(min_fig.get_figure())
    return None


def plot_cs_w_unembed(
    saes: List[sae_lens.SAE],
    model: sae_lens.HookedSAETransformer,
    model_name: str = "gemma2",
) -> None:
    cs_min_stats = []
    cs_max_stats = []
    layers, _, _ = name2lrc(model_name)
    for layer in range(layers):
        cs_max_stats.append(
            pd.DataFrame(
                {
                    "cos": get_cosine_similarity(saes[layer].W_dec, model.unembed.W_U.T)
                    .fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                }
            )
        )
        cs_min_stats.append(
            pd.DataFrame(
                {
                    "cos": get_cosine_similarity(saes[layer].W_dec, model.unembed.W_U.T)
                    .fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                }
            )
        )
    max_fig = sns.boxplot(pd.concat(cs_max_stats, axis=0), x="layer", y="cos")
    min_fig = sns.boxplot(pd.concat(cs_min_stats, axis=0), x="layer", y="cos")
    min_fig.get_figure().savefig(f"./res/cos_sim/{model_name}_cs_w_unembed.pdf")
    plt.close(max_fig.get_figure())
    plt.close(min_fig.get_figure())
    return None


def get_top_index(acts: torch.Tensor, top_num: int, layer: int) -> np.ndarray:
    code_acts = acts[1]
    math_acts = acts[0]
    wiki_acts = acts[2]
    top_index_code = torch.topk(code_acts[layer], top_num).indices
    top_index_math = torch.topk(math_acts[layer], top_num).indices
    top_index_wiki = torch.topk(wiki_acts[layer], top_num).indices
    top_index_mc = np.intersect1d(
        top_index_code.cpu().numpy(), top_index_math.cpu().numpy()
    )
    top_index_mw = np.intersect1d(
        top_index_math.cpu().numpy(), top_index_wiki.cpu().numpy()
    )
    top_index_cw = np.intersect1d(
        top_index_code.cpu().numpy(), top_index_wiki.cpu().numpy()
    )
    top_index = np.intersect1d(top_index_mc, top_index_mw)
    top_index_wiki = np.setdiff1d(
        top_index_wiki.cpu().numpy(), np.union1d(top_index_cw, top_index_mw)
    )
    top_index_math = np.setdiff1d(
        top_index_math.cpu().numpy(), np.union1d(top_index_mc, top_index_mw)
    )
    top_index_code = np.setdiff1d(
        top_index_code.cpu().numpy(), np.union1d(top_index_mc, top_index_cw)
    )
    top_index_mc = np.setdiff1d(top_index_mc, top_index)
    top_index_mw = np.setdiff1d(top_index_mw, top_index)
    top_index_cw = np.setdiff1d(top_index_cw, top_index)
    return (
        code_acts,
        math_acts,
        wiki_acts,
        top_index_code,
        top_index_math,
        top_index_wiki,
        top_index_cw,
        top_index_mc,
        top_index_mw,
        top_index,
    )


def plot_top_freq(
    acts: torch.Tensor,
    model_name: str = "gemma2",
) -> None:
    """
    Plot the top frequency of activations.
    """
    layers, row, col = name2lrc(model_name)
    top_num = acts[0].shape[1] // 100
    fig, axes = plt.subplots(row, col, figsize=(col * 10, row * 10))
    for layer in range(layers):
        ax = axes[layer // col, layer % col]
        (
            code_acts,
            math_acts,
            wiki_acts,
            top_index_code,
            top_index_math,
            top_index_wiki,
            top_index_cw,
            top_index_mc,
            top_index_mw,
            top_index,
        ) = get_top_index(acts, top_num, layer)
        torch_top_index = torch.cat(
            (
                torch.tensor(top_index_code, device=code_acts.device),
                torch.tensor(top_index_math, device=code_acts.device),
                torch.tensor(top_index_wiki, device=code_acts.device),
                torch.tensor(top_index_cw, device=code_acts.device),
                torch.tensor(top_index_mc, device=code_acts.device),
                torch.tensor(top_index_mw, device=code_acts.device),
                torch.tensor(top_index, device=code_acts.device),
            ),
            dim=0,
        )
        ax.set_title(f"Layer {layer}")
        sns.lineplot(
            data=code_acts[layer][torch_top_index.cpu().numpy()]
            .to(torch.float32)
            .numpy(),
            label="Code Freq",
            ax=ax,
        )
        sns.lineplot(
            data=math_acts[layer][torch_top_index.cpu().numpy()]
            .to(torch.float32)
            .numpy(),
            label="Math Freq",
            ax=ax,
        )
        sns.lineplot(
            data=wiki_acts[layer][torch_top_index.cpu().numpy()]
            .to(torch.float32)
            .numpy(),
            label="Wiki Freq",
            ax=ax,
        )
        ax.legend()
        boundary_text = ["Code", "Math", "Wiki", "WC", "MC", "MW", "MCW"]
        idx = 0
        for boundary in [
            top_index_code.shape[0],
            top_index_code.shape[0] + top_index_math.shape[0],
            top_index_code.shape[0] + top_index_math.shape[0] + top_index_wiki.shape[0],
            top_index_code.shape[0]
            + top_index_math.shape[0]
            + top_index_wiki.shape[0]
            + top_index_cw.shape[0],
            top_index_code.shape[0]
            + top_index_math.shape[0]
            + top_index_wiki.shape[0]
            + top_index_cw.shape[0]
            + top_index_mc.shape[0],
            top_index_code.shape[0]
            + top_index_math.shape[0]
            + top_index_wiki.shape[0]
            + top_index_cw.shape[0]
            + top_index_mc.shape[0]
            + top_index_mw.shape[0],
        ]:
            ax.axvline(x=boundary, color="r", linestyle="--")
            ax.text(
                boundary, ax.get_ylim()[1], boundary_text[idx], color="r", ha="right"
            )
            idx += 1
        ax.text(
            torch_top_index.shape[0],
            ax.get_ylim()[1],
            boundary_text[idx],
            color="r",
            ha="right",
        )
        ax.set_xlabel("Vector Index")
        ax.set_ylabel("Frequency Value")
    # plt.show()
    if model_name == "gemma2":
        fig.delaxes(axes[6, 3])
        fig.delaxes(axes[6, 2])
    fig.savefig(f"./res/freq/{model_name}_top_freqs.pdf")
    plt.close(fig)
    return None


def plot_freq2cs_boxplot(
    acts: torch.Tensor,
    saes: List[sae_lens.SAE],
    model: sae_lens.HookedSAETransformer,
    model_name: str = "gemma2",
) -> None:
    """
    TODO: fix the ugly code
    Plot the frequency to cosine similarity boxplot.
    """
    layers, row, col = name2lrc(model_name)
    wiki_stat = []
    math_stat = []
    code_stat = []
    mwc_stat = []
    mw_stat = []
    cw_stat = []
    mc_stat = []
    wiki_stat_min = []
    math_stat_min = []
    code_stat_min = []
    mwc_stat_min = []
    mw_stat_min = []
    cw_stat_min = []
    mc_stat_min = []

    wiki_stat_w_unembed = []
    math_stat_w_unembed = []
    code_stat_w_unembed = []
    mwc_stat_w_unembed = []
    mw_stat_w_unembed = []
    cw_stat_w_unembed = []
    mc_stat_w_unembed = []

    wiki_stat_w_unembed_min = []
    math_stat_w_unembed_min = []
    code_stat_w_unembed_min = []
    mwc_stat_w_unembed_min = []
    mw_stat_w_unembed_min = []
    cw_stat_w_unembed_min = []
    mc_stat_w_unembed_min = []

    top_num = acts[0].shape[1] // 100
    for layer in range(layers):
        (
            code_acts,
            math_acts,
            wiki_acts,
            top_index_code,
            top_index_math,
            top_index_wiki,
            top_index_cw,
            top_index_mc,
            top_index_mw,
            top_index,
        ) = get_top_index(acts, top_num, layer)

        wiki_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_wiki, :], saes[layer].W_dec[top_index_wiki, :]
        )
        math_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_math, :], saes[layer].W_dec[top_index_math, :]
        )
        code_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_code, :], saes[layer].W_dec[top_index_code, :]
        )
        mw_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_mw, :], saes[layer].W_dec[top_index_mw, :]
        )
        mc_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_mc, :], saes[layer].W_dec[top_index_mc, :]
        )
        cw_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_cw, :], saes[layer].W_dec[top_index_cw, :]
        )
        mwc_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index, :], saes[layer].W_dec[top_index, :]
        )
        wiki_stat.append(
            pd.DataFrame(
                {
                    "cos": wiki_cs.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "wiki",
                }
            )
        )
        math_stat.append(
            pd.DataFrame(
                {
                    "cos": math_cs.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "math",
                }
            )
        )
        code_stat.append(
            pd.DataFrame(
                {
                    "cos": code_cs.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "code",
                }
            )
        )
        mw_stat.append(
            pd.DataFrame(
                {
                    "cos": mw_cs.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mw",
                }
            )
        )
        mc_stat.append(
            pd.DataFrame(
                {
                    "cos": mc_cs.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mc",
                }
            )
        )
        cw_stat.append(
            pd.DataFrame(
                {
                    "cos": cw_cs.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "cw",
                }
            )
        )
        mwc_stat.append(
            pd.DataFrame(
                {
                    "cos": mwc_cs.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mwc",
                }
            )
        )

        wiki_stat_min.append(
            pd.DataFrame(
                {
                    "cos": wiki_cs.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "wiki_min",
                }
            )
        )
        math_stat_min.append(
            pd.DataFrame(
                {
                    "cos": math_cs.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "math_min",
                }
            )
        )
        code_stat_min.append(
            pd.DataFrame(
                {
                    "cos": code_cs.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "code_min",
                }
            )
        )
        mw_stat_min.append(
            pd.DataFrame(
                {
                    "cos": mw_cs.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mw_min",
                }
            )
        )
        mc_stat_min.append(
            pd.DataFrame(
                {
                    "cos": mc_cs.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mc_min",
                }
            )
        )
        cw_stat_min.append(
            pd.DataFrame(
                {
                    "cos": cw_cs.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "cw_min",
                }
            )
        )
        mwc_stat_min.append(
            pd.DataFrame(
                {
                    "cos": mwc_cs.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mwc_min",
                }
            )
        )

        wiki_cs_w_unembed = get_cosine_similarity(
            saes[layer].W_dec[top_index_wiki, :], model.unembed.W_U.T
        )
        math_cs_w_unembed = get_cosine_similarity(
            saes[layer].W_dec[top_index_math, :], model.unembed.W_U.T
        )
        code_cs_w_unembed = get_cosine_similarity(
            saes[layer].W_dec[top_index_code, :], model.unembed.W_U.T
        )
        mw_cs_w_unembed = get_cosine_similarity(
            saes[layer].W_dec[top_index_mw, :], model.unembed.W_U.T
        )
        mc_cs_w_unembed = get_cosine_similarity(
            saes[layer].W_dec[top_index_mc, :], model.unembed.W_U.T
        )
        cw_cs_w_unembed = get_cosine_similarity(
            saes[layer].W_dec[top_index_cw, :], model.unembed.W_U.T
        )
        mwc_cs_w_unembed = get_cosine_similarity(
            saes[layer].W_dec[top_index, :], model.unembed.W_U.T
        )
        wiki_stat_w_unembed.append(
            pd.DataFrame(
                {
                    "cos": wiki_cs_w_unembed.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "wiki",
                }
            )
        )
        math_stat_w_unembed.append(
            pd.DataFrame(
                {
                    "cos": math_cs_w_unembed.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "math",
                }
            )
        )
        code_stat_w_unembed.append(
            pd.DataFrame(
                {
                    "cos": code_cs_w_unembed.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "code",
                }
            )
        )
        mw_stat_w_unembed.append(
            pd.DataFrame(
                {
                    "cos": mw_cs_w_unembed.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mw",
                }
            )
        )
        mc_stat_w_unembed.append(
            pd.DataFrame(
                {
                    "cos": mc_cs_w_unembed.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mc",
                }
            )
        )
        cw_stat_w_unembed.append(
            pd.DataFrame(
                {
                    "cos": cw_cs_w_unembed.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "cw",
                }
            )
        )
        mwc_stat_w_unembed.append(
            pd.DataFrame(
                {
                    "cos": mwc_cs_w_unembed.fill_diagonal_(-100)
                    .max(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mwc",
                }
            )
        )
        wiki_stat_w_unembed_min.append(
            pd.DataFrame(
                {
                    "cos": wiki_cs_w_unembed.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "wiki_min",
                }
            )
        )
        math_stat_w_unembed_min.append(
            pd.DataFrame(
                {
                    "cos": math_cs_w_unembed.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "math_min",
                }
            )
        )
        code_stat_w_unembed_min.append(
            pd.DataFrame(
                {
                    "cos": code_cs_w_unembed.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "code_min",
                }
            )
        )
        mw_stat_w_unembed_min.append(
            pd.DataFrame(
                {
                    "cos": mw_cs_w_unembed.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mw_min",
                }
            )
        )
        mc_stat_w_unembed_min.append(
            pd.DataFrame(
                {
                    "cos": mc_cs_w_unembed.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mc_min",
                }
            )
        )
        cw_stat_w_unembed_min.append(
            pd.DataFrame(
                {
                    "cos": cw_cs_w_unembed.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "cw_min",
                }
            )
        )
        mwc_stat_w_unembed_min.append(
            pd.DataFrame(
                {
                    "cos": mwc_cs_w_unembed.fill_diagonal_(100)
                    .min(1)
                    .values.cpu()
                    .to(torch.float32)
                    .numpy(),
                    "layer": layer + 1,
                    "type": "mwc_min",
                }
            )
        )
    fig, axes = plt.subplots(2, 4, figsize=(40, 20))
    fig.delaxes(axes[1, 3])

    axes[0, 0].set_title("Wiki Cosine Similarity")
    axes[0, 1].set_title("Math Cosine Similarity")
    axes[0, 2].set_title("Code Cosine Similarity")
    axes[0, 3].set_title("MW Cosine Similarity")
    axes[1, 0].set_title("MC Cosine Similarity")
    axes[1, 1].set_title("CW Cosine Similarity")
    axes[1, 2].set_title("MWC Cosine Similarity")
    sns.boxplot(
        data=pd.concat(wiki_stat), x="layer", y="cos", hue="type", ax=axes[0, 0]
    )
    sns.boxplot(
        data=pd.concat(math_stat), x="layer", y="cos", hue="type", ax=axes[0, 1]
    )
    sns.boxplot(
        data=pd.concat(code_stat), x="layer", y="cos", hue="type", ax=axes[0, 2]
    )
    sns.boxplot(data=pd.concat(mw_stat), x="layer", y="cos", hue="type", ax=axes[0, 3])
    sns.boxplot(data=pd.concat(mc_stat), x="layer", y="cos", hue="type", ax=axes[1, 0])
    sns.boxplot(data=pd.concat(cw_stat), x="layer", y="cos", hue="type", ax=axes[1, 1])
    sns.boxplot(data=pd.concat(mwc_stat), x="layer", y="cos", hue="type", ax=axes[1, 2])
    sns.boxplot(
        data=pd.concat(wiki_stat_min), x="layer", y="cos", hue="type", ax=axes[0, 0]
    )
    sns.boxplot(
        data=pd.concat(math_stat_min), x="layer", y="cos", hue="type", ax=axes[0, 1]
    )
    sns.boxplot(
        data=pd.concat(code_stat_min), x="layer", y="cos", hue="type", ax=axes[0, 2]
    )
    sns.boxplot(
        data=pd.concat(mw_stat_min), x="layer", y="cos", hue="type", ax=axes[0, 3]
    )
    sns.boxplot(
        data=pd.concat(mc_stat_min), x="layer", y="cos", hue="type", ax=axes[1, 0]
    )
    sns.boxplot(
        data=pd.concat(cw_stat_min), x="layer", y="cos", hue="type", ax=axes[1, 1]
    )
    sns.boxplot(
        data=pd.concat(mwc_stat_min), x="layer", y="cos", hue="type", ax=axes[1, 2]
    )
    plt.tight_layout()
    fig.savefig(f"./res/cos_sim/{model_name}_top_freq_cs.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(40, 20))
    fig.delaxes(axes[1, 3])
    axes[0, 0].set_title("Wiki Cosine Similarity w/ Unembed")
    axes[0, 1].set_title("Math Cosine Similarity w/ Unembed")
    axes[0, 2].set_title("Code Cosine Similarity w/ Unembed")
    axes[0, 3].set_title("MW Cosine Similarity w/ Unembed")
    axes[1, 0].set_title("MC Cosine Similarity w/ Unembed")
    axes[1, 1].set_title("CW Cosine Similarity w/ Unembed")
    axes[1, 2].set_title("MWC Cosine Similarity w/ Unembed")
    sns.boxplot(
        data=pd.concat(wiki_stat_w_unembed),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[0, 0],
    )
    sns.boxplot(
        data=pd.concat(math_stat_w_unembed),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[0, 1],
    )
    sns.boxplot(
        data=pd.concat(code_stat_w_unembed),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[0, 2],
    )
    sns.boxplot(
        data=pd.concat(mw_stat_w_unembed), x="layer", y="cos", hue="type", ax=axes[0, 3]
    )
    sns.boxplot(
        data=pd.concat(mc_stat_w_unembed), x="layer", y="cos", hue="type", ax=axes[1, 0]
    )
    sns.boxplot(
        data=pd.concat(cw_stat_w_unembed), x="layer", y="cos", hue="type", ax=axes[1, 1]
    )
    sns.boxplot(
        data=pd.concat(mwc_stat_w_unembed),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[1, 2],
    )
    sns.boxplot(
        data=pd.concat(wiki_stat_w_unembed_min),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[0, 0],
    )
    sns.boxplot(
        data=pd.concat(math_stat_w_unembed_min),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[0, 1],
    )
    sns.boxplot(
        data=pd.concat(code_stat_w_unembed_min),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[0, 2],
    )
    sns.boxplot(
        data=pd.concat(mw_stat_w_unembed_min),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[0, 3],
    )
    sns.boxplot(
        data=pd.concat(mc_stat_w_unembed_min),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[1, 0],
    )
    sns.boxplot(
        data=pd.concat(cw_stat_w_unembed_min),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[1, 1],
    )
    sns.boxplot(
        data=pd.concat(mwc_stat_w_unembed_min),
        x="layer",
        y="cos",
        hue="type",
        ax=axes[1, 2],
    )
    plt.tight_layout()
    fig.savefig(f"./res/cos_sim/{model_name}_top_freq_cs_w_unembed.pdf")
    plt.close(fig)

    return None


# TODO: model and model_name seems repeated
def plot_freq2cs_lineplot(
    acts: torch.Tensor,
    saes: List[sae_lens.SAE],
    model_name: str = "gemma2",
    max_cs: bool = True,
) -> None:
    """
    Plot the inter-intra math, code, wiki, common / unembedded matrix
    """
    layers, row, col = name2lrc(model_name)
    line_name = [
        "wiki",
        "math",
        "code",
        "common",
        "wcc",
        "ccc",
        "mcc",
        "wm",
        "cm",
        "wc",
        "overall",
    ]
    stats = [[] for _ in range(len(line_name))]
    top_num = acts[0].shape[1] // 100
    for layer in range(layers):
        (
            code_acts,
            math_acts,
            wiki_acts,
            top_index_code,
            top_index_math,
            top_index_wiki,
            top_index_cw,
            top_index_mc,
            top_index_mw,
            top_index,
        ) = get_top_index(acts, top_num, layer)

        wiki_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_wiki, :], saes[layer].W_dec[top_index_wiki, :]
        )
        math_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_math, :], saes[layer].W_dec[top_index_math, :]
        )
        code_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_code, :], saes[layer].W_dec[top_index_code, :]
        )
        common_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index, :], saes[layer].W_dec[top_index, :]
        )
        wcc_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_wiki, :], saes[layer].W_dec[top_index, :]
        )
        ccc_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_code, :], saes[layer].W_dec[top_index, :]
        )
        mcc_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_math, :], saes[layer].W_dec[top_index, :]
        )
        wm_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_wiki, :], saes[layer].W_dec[top_index_math, :]
        )
        cm_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_code, :], saes[layer].W_dec[top_index_math, :]
        )
        wc_cs = get_cosine_similarity(
            saes[layer].W_dec[top_index_wiki, :], saes[layer].W_dec[top_index_code, :]
        )
        overall_cs = get_cosine_similarity(saes[layer].W_dec, saes[layer].W_dec)
        for idx, cs in enumerate(
            [
                wiki_cs,
                math_cs,
                code_cs,
                common_cs,
                wcc_cs,
                ccc_cs,
                mcc_cs,
                wm_cs,
                cm_cs,
                wc_cs,
                overall_cs,
            ]
        ):
            if max_cs:
                stats[idx].append(
                    float(
                        cs.fill_diagonal_(-100).max(1).values.mean().to(torch.float32)
                    )
                )
            else:
                stats[idx].append(
                    float(
                        cs.fill_diagonal_(0).abs().sum().to(torch.float32)
                        / (cs.shape[0] * (cs.shape[1] - 1))
                    )
                )

    stat_list_df = []
    for idx, stat in enumerate(stats):
        stat_df = pd.DataFrame(
            {
                "cos": stat,
                "layer": list(range(1, layers + 1)),
                "line_name": [line_name[idx]] * layers,
            }
        )
        stat_list_df.append(stat_df)
    stat_df = pd.concat(stat_list_df)
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(data=stat_df, x="layer", y="cos", hue="line_name", ax=ax)
    ax.set_title("Average Cosine Similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Cosine Similarity")
    if max_cs:
        plt.savefig(f"./res/cos_sim/max_{model_name}_freq2cs_lineplot.pdf")
    else:
        plt.savefig(f"./res/cos_sim/avg_{model_name}_freq2cs_lineplot.pdf")
    plt.close()
    return None


def plot_dataset_geometry(
    acts: torch.Tensor,
    saes: List[sae_lens.SAE],
    model: sae_lens.HookedSAETransformer,
    model_name: str = "gemma2",
) -> None:
    """
    Plot the dataset geometry.
    """
    plot_top_freq(acts, model_name=model_name)
    plot_freq2cs_boxplot(acts, model_name=model_name, saes=saes, model=model)
    plot_freq2cs_lineplot(acts, model_name=model_name, saes=saes)
    return None


def plot_basic_geometry(
    acts: torch.Tensor,
    saes: List[sae_lens.SAE],
    model: sae_lens.HookedSAETransformer,
    model_name: str = "gemma2",
    data_name: List = ["math", "code", "wiki"],
) -> None:
    """
    Plot the basic geometry.
    """
    plot_freq(acts, model_name=model_name, data_name=data_name)
    plot_cs(model_name=model_name, saes=saes)
    plot_cs_w_unembed(model_name=model_name, saes=saes, model=model)
    return None


@torch.no_grad()
def ablation_decoder(
    acts: torch.Tensor,
    model: sae_lens.HookedSAETransformer,
    model_name: str = "gemma2",
    ds: List = ["wiki", "code", "math"],
    use_error_term: bool = False,
) -> None:
    """
    1. load data
    2. ablate sae
    3. run model to see its influence on output and next layer
    """
    # 1. load data
    for data_name in ds:
        if data_name == "wiki":
            dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
                "train"
            ]
            text = "text"
            ds_ratio = 3e-3
        elif data_name == "code":
            dataset = datasets.load_dataset("b-mc2/sql-create-context")["train"]
            text = "answer"
            ds_ratio = 1e-3
        elif data_name == "math":
            dataset = datasets.load_dataset("TIGER-Lab/MathInstruct")["train"]
            text = "output"
            ds_ratio = 5e-4
        # 2. ablate sae
        layers, _, _ = name2lrc(model_name)
        layers = 2
        top_num = acts[0].shape[1] // 10
        abl_times = 1
        abl_num = top_num // 10
        length_ds = int(len(dataset) * ds_ratio)
        # running times: layers * vector_group * abl_times * length_ds
        for layer in range(layers - 1):
            (
                _,
                _,
                _,
                top_index_code,
                top_index_math,
                top_index_wiki,
                top_index_cw,
                top_index_mc,
                top_index_mw,
                top_index,
            ) = get_top_index(acts, top_num, layer)
            name = ["code", "math", "wiki", "cw", "mc", "mw", "mcw"]
            name_idx = 0
            if model_name == "gemma2":
                release = "gemma-scope-2b-pt-res-canonical"
                sae_id = f"layer_{layer}/width_16k/canonical"
                saes = []
                saes.append(
                    sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0].to(
                        dtype=torch.bfloat16
                    )
                )
                sae_id = f"layer_{layer+1}/width_16k/canonical"
                saes.append(
                    sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0].to(
                        dtype=torch.bfloat16
                    )
                )
            elif model_name == "llama3":
                release = "llama_scope_lxr_8x"
                saes = []
                sae_id = f"l{layer}r_8x"
                saes.append(
                    sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0]
                )
                sae_id = f"l{layer+1}r_8x"
                saes.append(
                    sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0]
                )
            else:
                release = "pythia-70m-deduped-res-sm"
                saes = []
                sae_id = f"blocks.{layer}.hook_resid_post"
                saes.append(
                    sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0].to(
                        dtype=torch.bfloat16
                    )
                )
                sae_id = f"blocks.{layer+1}.hook_resid_post"
                saes.append(
                    sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0].to(
                        dtype=torch.bfloat16
                    )
                )
            for top_t in tqdm([
                top_index_code,
                top_index_math,
                top_index_wiki,
                top_index_cw,
                top_index_mc,
                top_index_mw,
                top_index,
            ]):
                for idx in range(abl_times):
                    if model_name == "gemma2":
                        release = "gemma-scope-2b-pt-res-canonical"
                        sae_id = f"layer_{layer}/width_16k/canonical"
                        saes2 = []
                        saes2.append(
                            sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[
                                0
                            ].to(dtype=torch.bfloat16)
                        )
                        sae_id = f"layer_{layer+1}/width_16k/canonical"
                        saes2.append(
                            sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[
                                0
                            ].to(dtype=torch.bfloat16)
                        )
                    elif model_name == "llama3":
                        release = "llama_scope_lxr_8x"
                        saes2 = []
                        sae_id = f"l{layer}r_8x"
                        saes2.append(
                            sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0].to(dtype=torch.bfloat16)
                        )
                        sae_id = f"l{layer+1}r_8x"
                        saes2.append(
                            sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0].to(dtype=torch.bfloat16)
                        )
                    list(
                        map(
                            lambda idy: saes2[0].W_dec[idy, :].zero_(),
                            top_t[abl_num * idx : abl_num * (idx + 1)],
                        )
                    )
                    doc_len = 0
                    freqs = torch.zeros(saes[0].cfg.d_sae)
                    loss = torch.zeros(length_ds)
                    for idy in tqdm(range(length_ds)):
                        # loop begin, fuck indent
                        example = dataset[idy]
                        tokens = model.to_tokens([example[text]], prepend_bos=True)
                        loss1, cache1 = model.run_with_cache_with_saes(
                            tokens, saes=saes, use_error_term=use_error_term
                        )
                        model.reset_saes()
                        loss2, cache2 = model.run_with_cache_with_saes(
                            tokens, saes=saes2, use_error_term=use_error_term
                        )
                        model.reset_saes()
                        local_doc_len = cache1[
                            f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
                        ].shape[1]
                        freq = torch.zeros_like(freqs)

                        prompt2 = f"blocks.{layer+1}.hook_resid_post.hook_sae_acts_post"
                        freq = (
                            abs(
                                (
                                    (cache1[prompt2] > 1e-3)
                                    + 0
                                    + (cache2[prompt2] > 1e-3)
                                    - 1
                                )
                            )
                            < 1e-2
                        )[0].sum(0) / local_doc_len
                        loss[idy] = (loss1 - loss2).sum().item()
                        # freq[layer] = (cache[prompt2] > 1e-3)[0].sum(0) / local_doc_len
                        new_doc_len = doc_len + local_doc_len
                        if idy == 0:
                            freqs = freq
                        else:
                            freqs = (
                                freqs * doc_len / new_doc_len
                                + freq * local_doc_len / new_doc_len
                            )
                        doc_len = new_doc_len
                    torch.save(
                        freqs,
                        f"./res/acts/abl/{model_name}_{data_name}_layer{layer}_abl{idx}_top{name[name_idx]}.pt",
                    )
                    torch.save(
                        loss,
                        f"./res/acts/abl/{model_name}_{data_name}_layer{layer}_abl{idx}_loss_top{name[name_idx]}.pt",
                    )
                name_idx += 1


def ablation_load_save(
    abl_path: str = "./res/acts/abl",
    model_name: str = "llama3",
    data_name: List = ["math", "code", "wiki"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return: torch.tensor [ds, layer, abl_times, abl_group, sae_size], torch.tensor [ds, layer, abl_times, abl_group, length_ds]"""
    layers, _, _ = name2lrc(model_name)
    abl_times = 5
    for data in data_name:
        for layer in range(layers - 1):
            for idx in range(abl_times):
                for name in ["code", "math", "wiki", "cw", "mc", "mw", "mcw"]:
                    freq = torch.load(
                        f"{abl_path}/{model_name}_{data}_layer{layer}_abl{idx}_top{name}.pt"
                    )
                    loss = torch.load(
                        f"{abl_path}/{model_name}_{data}_layer{layer}_abl{idx}_loss_top{name}.pt"
                    )
                    print(freq, loss)


def draw_IOU_ablation(
    acts: List[torch.tensor],
    abl_path: str = "./res/acts/abl",
    model_name: str = "llama3",
    iou_number: bool = False,
):
    layers, row, col = name2lrc(model_name)
    abl_times = 5
    vector_group = ["code", "math", "wiki", "cw", "mc", "mw", "mcw"]
    layer_index = [
        [[] for _ in vector_group] for i in range(layers - 1)
    ]  # [layer][abl_group]
    if model_name == "gemma2":
        d_sae = 16384
    else:
        d_sae = 32768
    top_num = acts[0].shape[1] // 10
    for layer in range(layers - 1):
        (
            code_acts,
            math_acts,
            wiki_acts,
            top_index_code,
            top_index_math,
            top_index_wiki,
            top_index_cw,
            top_index_mc,
            top_index_mw,
            top_index,
        ) = get_top_index(acts, top_num, layer)
        layer_index[layer][0] = top_index_code
        layer_index[layer][1] = top_index_math
        layer_index[layer][2] = top_index_wiki
        layer_index[layer][3] = top_index_cw
        layer_index[layer][4] = top_index_mc
        layer_index[layer][5] = top_index_mw
        layer_index[layer][6] = top_index
    for data in ["math", "code", "wiki"]:
        for ablation in tqdm(["code", "math", "wiki", "cw", "mc", "mw", "mcw"]):
            fig, axes = plt.subplots(2, 4, figsize=(40, 20))
            freq = torch.zeros(layers - 1, abl_times, d_sae)
            # freq = torch.load(
            #     osp.join('./res/abl_freq', f"{model_name}_{data}_abl_{ablation}_freq.pt")
            # )
            wiki_res_shape = [[[] for i in range(abl_times)] for _ in range(layers - 1)]
            math_res_shape = [[[] for i in range(abl_times)] for _ in range(layers - 1)]
            code_res_shape = [[[] for i in range(abl_times)] for _ in range(layers - 1)]
            mw_res_shape = [[[] for i in range(abl_times)] for _ in range(layers - 1)]
            mc_res_shape = [[[] for i in range(abl_times)] for _ in range(layers - 1)]
            cw_res_shape = [[[] for i in range(abl_times)] for _ in range(layers - 1)]
            common_res_shape = [
                [[] for i in range(abl_times)] for _ in range(layers - 1)
            ]
            for layer in range(layers - 1):
                for idx in range(abl_times):
                    freq_name = (
                        f"{model_name}_{data}_layer{layer}_abl{idx}_top{ablation}.pt"
                    )
                    path = osp.join(abl_path, freq_name)
                    freq[layer, idx, :] = torch.load(path)
                    code_res_shape[layer][idx] = np.intersect1d(
                        layer_index[layer][0],
                        freq[layer, idx, :].nonzero().cpu().numpy(),
                    )
                    math_res_shape[layer][idx] = np.intersect1d(
                        layer_index[layer][1],
                        freq[layer, idx, :].nonzero().cpu().numpy(),
                    )
                    wiki_res_shape[layer][idx] = np.intersect1d(
                        layer_index[layer][2],
                        freq[layer, idx, :].nonzero().cpu().numpy(),
                    )
                    cw_res_shape[layer][idx] = np.intersect1d(
                        layer_index[layer][3],
                        freq[layer, idx, :].nonzero().cpu().numpy(),
                    )
                    mc_res_shape[layer][idx] = np.intersect1d(
                        layer_index[layer][4],
                        freq[layer, idx, :].nonzero().cpu().numpy(),
                    )
                    mw_res_shape[layer][idx] = np.intersect1d(
                        layer_index[layer][5],
                        freq[layer, idx, :].nonzero().cpu().numpy(),
                    )
                    common_res_shape[layer][idx] = np.intersect1d(
                        layer_index[layer][6],
                        freq[layer, idx, :].nonzero().cpu().numpy(),
                    )
            if iou_number:
                axes[0, 0].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on wiki"
                )
                for layer in range(layers - 1):
                    axes[0, 0].boxplot(
                        [len(wiki_res_shape[layer][idx]) for idx in range(abl_times)],
                        positions=[layer + 1],
                    )
                axes[0, 1].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on math"
                )
                for layer in range(layers - 1):
                    axes[0, 1].boxplot(
                        [len(math_res_shape[layer][idx]) for idx in range(abl_times)],
                        positions=[layer + 1],
                    )
                axes[0, 2].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on code"
                )
                for layer in range(layers - 1):
                    axes[0, 2].boxplot(
                        [len(code_res_shape[layer][idx]) for idx in range(abl_times)],
                        positions=[layer + 1],
                    )
                axes[0, 3].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on mw"
                )
                for layer in range(layers - 1):
                    axes[0, 3].boxplot(
                        [len(mw_res_shape[layer][idx]) for idx in range(abl_times)],
                        positions=[layer + 1],
                    )
                axes[1, 0].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on mc"
                )
                for layer in range(layers - 1):
                    axes[1, 0].boxplot(
                        [len(mc_res_shape[layer][idx]) for idx in range(abl_times)],
                        positions=[layer + 1],
                    )
                axes[1, 1].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on cw"
                )
                for layer in range(layers - 1):
                    axes[1, 1].boxplot(
                        [len(cw_res_shape[layer][idx]) for idx in range(abl_times)],
                        positions=[layer + 1],
                    )
                axes[1, 2].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on mcw"
                )
                for layer in range(layers - 1):
                    axes[1, 2].boxplot(
                        [len(common_res_shape[layer][idx]) for idx in range(abl_times)],
                        positions=[layer + 1],
                    )
                plt.tight_layout()
                fig.delaxes(axes[1, 3])
                fig.savefig(
                    f"./res/abl_freq_res/{model_name}_{data}_abl_{ablation}_freq_shape.pdf"
                )
            else:
                axes[0, 0].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on wiki"
                )
                for layer in range(layers - 1):
                    axes[0, 0].boxplot(
                        [
                            freq[layer, idx, wiki_res_shape[layer][idx]]
                            .mean()
                            .cpu()
                            .numpy()
                            for idx in range(abl_times)
                        ],
                        positions=[layer + 1],
                    )
                axes[0, 1].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on math"
                )
                for layer in range(layers - 1):
                    axes[0, 1].boxplot(
                        [
                            freq[layer, idx, math_res_shape[layer][idx]]
                            .mean()
                            .cpu()
                            .numpy()
                            for idx in range(abl_times)
                        ],
                        positions=[layer + 1],
                    )
                axes[0, 2].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on code"
                )
                for layer in range(layers - 1):
                    axes[0, 2].boxplot(
                        [
                            freq[layer, idx, code_res_shape[layer][idx]]
                            .mean()
                            .cpu()
                            .numpy()
                            for idx in range(abl_times)
                        ],
                        positions=[layer + 1],
                    )
                axes[0, 3].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on mw"
                )
                for layer in range(layers - 1):
                    axes[0, 3].boxplot(
                        [
                            freq[layer, idx, mw_res_shape[layer][idx]]
                            .mean()
                            .cpu()
                            .numpy()
                            for idx in range(abl_times)
                        ],
                        positions=[layer + 1],
                    )
                axes[1, 0].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on mc"
                )
                for layer in range(layers - 1):
                    axes[1, 0].boxplot(
                        [
                            freq[layer, idx, mc_res_shape[layer][idx]]
                            .mean()
                            .cpu()
                            .numpy()
                            for idx in range(abl_times)
                        ],
                        positions=[layer + 1],
                    )
                axes[1, 1].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on cw"
                )
                for layer in range(layers - 1):
                    axes[1, 1].boxplot(
                        [
                            freq[layer, idx, cw_res_shape[layer][idx]]
                            .mean()
                            .cpu()
                            .numpy()
                            for idx in range(abl_times)
                        ],
                        positions=[layer + 1],
                    )
                axes[1, 2].set_title(
                    f"Ablate {ablation} on {data} dataset, influence on mcw"
                )
                for layer in range(layers - 1):
                    axes[1, 2].boxplot(
                        [
                            freq[layer, idx, common_res_shape[layer][idx]]
                            .mean()
                            .cpu()
                            .numpy()
                            for idx in range(abl_times)
                        ],
                        positions=[layer + 1],
                    )
                plt.tight_layout()
                fig.delaxes(axes[1, 3])
                fig.savefig(
                    f"./res/abl_freq_res/{model_name}_{data}_abl_{ablation}_freq_value.pdf"
                )
            torch.save(
                freq, f"./res/abl_freq/{model_name}_{data}_abl_{ablation}_freq.pt"
            )
            plt.close(fig)


def ablation_loss(abl_path: str = "./res/acts/abl", model_name: str = "llama3"):
    layers, _, _ = name2lrc(model_name)
    vector_group = ["code", "math", "wiki", "cw", "mc", "mw", "mcw"]
    dataset_name = ["math", "code", "wiki"]
    abl_times = 5
    loss = torch.zeros(layers - 1, len(dataset_name), len(vector_group), abl_times)
    for layer in range(layers - 1):
        for data in dataset_name:
            for ablation in vector_group:
                for idx in range(abl_times):
                    freq_name = f"{model_name}_{data}_layer{layer}_abl{idx}_loss_top{ablation}.pt"
                    path = osp.join(abl_path, freq_name)
                    ablation_idx = vector_group.index(ablation)
                    dataset_name_idx = dataset_name.index(data)
                    loss[layer, dataset_name_idx, ablation_idx, idx] = torch.load(path).mean()
    torch.save(loss, f"./res/abl_loss/{model_name}_loss.pt")
    fig, axes = plt.subplots(7, 3, figsize=(30, 10))
    for idx, data in enumerate(dataset_name):
        for idy, ablation in enumerate(vector_group):
            axes[idy, idx].set_title(f"{model_name}: ablate {ablation} on {data} dataset loss")
            sns.boxplot(data=loss[:, idx, idy, :], ax=axes[idy, idx])
    plt.tight_layout()
    fig.savefig(f"./res/abl_loss/{model_name}_loss.pdf")
    plt.close(fig)
    return None
                


if __name__ == "__main__":
    args = config()
    set_seed(args.seed)
    # if args.model_name == "gemma2":
    #     saes, model = obtain_gemma_data()
    #     layers = 26
    # elif args.model_name == "llama3":
    #     saes, model = obtain_llama_data()
    #     layers = 32
    # else:
    #     saes, model = obtain_pythia_data()
    #     layers = 6
    # # if args.model_name == "gemma2":
    # #     model_name = "gemma-2-2b"
    # #     model = sae_lens.HookedSAETransformer.from_pretrained(
    # #         model_name, dtype=torch.bfloat16
    # #     )
    # # elif args.model_name == "llama3":
    # #     model_name = "meta-llama/Llama-3.1-8B"
    # #     model = sae_lens.HookedSAETransformer.from_pretrained(
    # #         model_name, dtype=torch.bfloat16
    # #     )
    # # else:
    # #     model_name = "EleutherAI/pythia-70m-deduped"
    # #     model = sae_lens.HookedSAETransformer.from_pretrained(
    # #         model_name, dtype=torch.bfloat16
    # #     )
    # # saes, model = obtain_gemma_data()
    # # saes, model = obtain_llama_data()
    # torch.cuda.empty_cache()
    # # acts = obtain_acts(saes, model, layers, model_name=args.model_name)
    # acts = load_acts_from_pretrained(model_name=args.model_name)
    # plot_basic_geometry(acts, model_name=args.model_name, saes=saes, model=model)
    # plot_dataset_geometry(acts, model_name=args.model_name, saes=saes, model=model)
    # # ablation_decoder(
    # #     acts,
    # #     model,
    # #     model_name=args.model_name,
    # #     use_error_term=args.use_error_term,
    # # )
    if args.model_name == "gemma2":
        model_name = "gemma-2-2b"
        model = sae_lens.HookedSAETransformer.from_pretrained(
            model_name, dtype=torch.bfloat16
        )
    elif args.model_name == "llama3":
        model_name = "meta-llama/Llama-3.1-8B"
        model = sae_lens.HookedSAETransformer.from_pretrained(
            model_name, dtype=torch.bfloat16
        )
    acts = load_acts_from_pretrained(model_name=args.model_name)
    ablation_decoder(
        acts,
        model,
        model_name=args.model_name,
        use_error_term=args.use_error_term,
    )
    # draw_IOU_ablation(acts, model_name=args.model_name)
    # ablation_loss(model_name=args.model_name)