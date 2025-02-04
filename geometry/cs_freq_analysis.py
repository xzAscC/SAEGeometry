import torch
import jaxtyping
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
    for layer, sae in enumerate(saes):
        dataset_length = int(len(dataset))
        cs = get_cosine_similarity(sae.W_dec, sae.W_dec).fill_diagonal_(-100).cpu()
        max_indices = torch.where(cs > 0.9)
        min_indices = torch.where((cs > -1.1) & (cs < -0.9))
        zero_indices = torch.where((cs > -0.1) & (cs < 0.1))
        cs_pair_num = 100
        for idx in tqdm(range(dataset_length)):
            example = dataset[idx]
            tokens = model.to_tokens([example[text]], prepend_bos=True)
            loss1, cache1 = model.run_with_cache_with_saes(
                tokens, saes=sae, use_error_term=False
            )
            model.reset_saes()
            if model_name == "pythia-70m-deduped":
                prompt = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
            dim1 = cache1[prompt].shape[1]
            for idy in range(cs_pair_num):
                try:
                    res1 = cache1[prompt][:, :, max_indices[0][idy]] > 1e-4
                    res2 = cache1[prompt][:, :, max_indices[1][idy]] > 1e-4
                    if res1.sum() == 0 and res2.sum() == 0:
                        pass
                    else:
                        max_diffs[layer].append(((res1 & res2).sum() / dim1).cpu().item())
                except:
                    pass
                try:
                    res1 = cache1[prompt][:, :, min_indices[0][idy]] > 1e-4
                    res2 = cache1[prompt][:, :, min_indices[1][idy]] > 1e-4
                    if res1.sum() == 0 and res2.sum() == 0:
                        pass
                    else:
                        min_diffs[layer].append(((res1 & res2).sum() / dim1).cpu().item())
                except:
                    pass
                try:
                    res1 = cache1[prompt][:, :, zero_indices[0][idy]] > 1e-4
                    res2 = cache1[prompt][:, :, zero_indices[1][idy]] > 1e-4
                    if res1.sum() == 0 and res2.sum() == 0:
                        pass
                    else:
                        zero_diffs[layer].append(((res1 & res2).sum() / dim1).cpu().item())
                except:
                    pass

        # TODO: plot the box plot here
        # sns.boxplot(min_diffs)
        # plt.title("Opposite cos sim vector frequency pattern Boxplot")
        # sns.boxplot(max_diffs)
        # plt.title("Nearly same cos sim vector frequency pattern Boxplot")
        # sns.boxplot(zero_diffs)
        # plt.title("Nearly Zero Cosine Similarity Pair Frequency Pattern")
        return max_diffs, min_diffs, zero_diffs


if __name__ == "__main__":
    max_diffs, min_diffs, zero_diffs = cs_freqPattern_comparsion(
        "pythia-70m-deduped", "MMLU"
    )
