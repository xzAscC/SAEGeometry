import random
import torch
import sae_lens
import numpy as np
from typing import List, Tuple, Dict, Any
from datasets import load_dataset

MAIN = "__main__"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sae_from_sae_lens(
    release: str, layers: int, device: str
) -> Dict[str, sae_lens.SAE]:
    sae_hook_to_name = {}
    for layer in range(layers):
        sae_id = f"blocks.{layer}.hook_resid_pre"
        sae_hook_to_name[sae_id] = sae_lens.SAE.from_pretrained(
            release, sae_id, device=device
        )[0]

    sae_id = "blocks.11.hook_resid_post"
    sae_hook_to_name[sae_id] = sae_lens.SAE.from_pretrained(
        release, sae_id, device=device
    )[0]
    return sae_hook_to_name


def load_LLM_from_transformers_lens(model_name: str) -> Any:
    return torch.load(model_name)


if MAIN:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    sae = load_sae_from_sae_lens("gpt2-small-res-jb", 12, device)
    dataset = load_dataset('Skylion007/openwebtext')
    model = sae_lens.HookedSAETransformer.from_pretrained("gpt2-small").to(device)

    dataset_length = int(0.001 * len(dataset['train']))
    print('finish loading dataset')
