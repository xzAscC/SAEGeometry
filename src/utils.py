import jaxtyping
import torch

@torch.no_grad()
def get_cosine_similarity(
    dict_elements_1: jaxtyping.Float[torch.Tensor, "n_dict_1 n_dense"],
    dict_elements_2: jaxtyping.Float[torch.Tensor, "n_dict_2 n_dense"],
    p_norm: int = 2,
    dim: int = 1,
) -> jaxtyping.Float[torch.Tensor, "n_dict_1 n_dict_2"]:
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
    dict_elements_1 = torch.nn.functional.normalize(dict_elements_1, p=p_norm, dim=dim)
    dict_elements_2 = torch.nn.functional.normalize(dict_elements_2, p=p_norm, dim=dim)

    # Compute cosine similarity using matrix multiplication
    cosine_sim: jaxtyping.Float[torch.Tensor, "n_dict_1 n_dict_2"] = torch.mm(dict_elements_1, dict_elements_2.T)
    # max_cosine_sim, _ = torch.max(cosine_sim, dim=1)
    return cosine_sim