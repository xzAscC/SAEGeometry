import sae_lens
import torch
import jaxtyping
import random
import datasets
import plotly.colors as pc
import plotly.express as px
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm


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


def set_seed(seed: int) -> None:
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


def obtain_data() -> (
    Tuple[List[sae_lens.SAE], torch.nn.Module, List[torch.utils.data.Dataset]]
):
    """
    load sae, model and dataset
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = 6
    saes = []
    release = "pythia-70m-deduped-res-sm"
    model_name = "pythia-70m-deduped"
    for layer in tqdm(range(layers)):
        sae_id = f"blocks.{layer}.hook_resid_post"
        saes.append(
            sae_lens.SAE.from_pretrained(release=release, sae_id=sae_id, device=device)[
                0
            ]
        )

    model = sae_lens.HookedSAETransformer.from_pretrained(model_name)
    ds = [datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]]

    return saes, model, ds


def plot_cosine_similarity(
    saes: List[sae_lens.SAE],
    model: torch.nn.ModuleList,
    is_umbedding: bool = False,
    self: bool = False,
) -> None:
    # Get the cosine similarity between the dictionary elements
    colors = pc.n_colors("rgb(5, 200, 200)", "rgb(200, 10, 10)", 13, colortype="rgb")
    min_cos_sim_stats = []
    max_cos_sim_stats = []
    for layer in range(len(saes)):
        if is_umbedding:
            cos_sim = get_cosine_similarity(saes[layer].W_dec, model.unembed.W_U.T)
        elif self:
            cos_sim = get_cosine_similarity(saes[layer].W_dec, saes[layer].W_dec)
        else:
            cos_sim = get_cosine_similarity(saes[layer].W_dec, saes[layer + 1].W_dec)

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

        if not is_umbedding and not self and layer == len(saes) - 2:
            break
    if is_umbedding:
        max_fig = sns.boxplot(
            data=pd.concat(max_cos_sim_stats, axis=0),
            x="layer",
            y="cos",
        )
        min_fig = sns.boxplot(
            data=pd.concat(min_cos_sim_stats, axis=0),
            x="layer",
            y="cos",
        )
        min_fig.get_figure().savefig("./res/cos_sim/vector_unembed_cs.pdf")
    elif self:
        # max_fig = px.box(
        #     pd.concat(max_cos_sim_stats, axis=0),
        #     x="layer",
        #     y="cos",
        #     width=800,
        #     height=600,
        #     color_discrete_sequence=colors,
        #     labels={"cos": "Max Cosine Similarity", "layer": "Layer"},
        # ).write_html("./res/cos_sim/max_cosine_similarity.html")
        # min_fig = px.box(
        #     pd.concat(min_cos_sim_stats, axis=0),
        #     x="layer",
        #     y="cos",
        #     width=800,
        #     height=600,
        #     color_discrete_sequence=colors,
        #     labels={"cos": "Min Cosine Similarity", "layer": "Layer"},
        # ).write_html("./res/cos_sim/min_cosine_similarity.html")
        max_fig = sns.boxplot(
            data=pd.concat(max_cos_sim_stats, axis=0),
            x="layer",
            y="cos",
        )
        min_fig = sns.boxplot(
            data=pd.concat(min_cos_sim_stats, axis=0),
            x="layer",
            y="cos",
        )
        min_fig.get_figure().savefig("./res/cos_sim/vector_vector_cs.pdf")
    else:
        max_fig = sns.boxplot(
            data=pd.concat(max_cos_sim_stats, axis=0),
            x="layer",
            y="cos",
        )
        min_fig = sns.boxplot(
            data=pd.concat(min_cos_sim_stats, axis=0),
            x="layer",
            y="cos",
        )
        min_fig.set_xlabel("Layer", fontsize=14)
        min_fig.set_ylabel("Cosine Similarity", fontsize=13)
        min_fig.tick_params(axis="both", which="major", labelsize=12)
        min_fig.get_figure().savefig("./res/cos_sim/vector_nvector_cs.pdf")
    return None


if __name__ == "__main__":
    set_seed(42)
    saes, model, ds = obtain_data()
    plot_cosine_similarity(saes, model, is_umbedding=False, self=False)
    acts = obtain_activations(saes, model, ds)
