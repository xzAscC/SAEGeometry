from typing import Tuple, List
import datasets
import sae_lens
import torch
from tqdm import tqdm


def fetch_sae_and_model(
    sae_name: str,
) -> Tuple[List[sae_lens.SAE], sae_lens.HookedSAETransformer]:
    """fetch the specified SAE and model given the SAE name

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
    for layer in range(layers):
        if release == "gemma-scope-2b-pt-res-canonical":
            sae_id = f"layer_{layer}/width_16k/canonical"
        elif release == "llama_scope_lxr_8x":
            sae_id = f"l{layer}r_8x"
        elif release == "pythia-70m-deduped-res-sm":
            sae_id = f"blocks.{layer}.hook_resid_post"
        sae = sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0]
        saes.append(sae)
    return saes


def fetch_model(model_name: str) -> sae_lens.HookedSAETransformer:
    model = sae_lens.HookedSAETransformer.from_pretrained_no_processing(
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
    elif dataset_name == "Math":
        dataset = datasets.load_dataset("hbin0701/abstract_math")["train"]
        text = "num_input"
        ratio = 5e-3
    elif dataset_name == "MMLU":
        dataset = datasets.load_dataset("cais/mmlu", "all")["validation"]
        ratio = 1e-2
        text = "question"
    return dataset, ratio, text

model_name = "gemma-2-2b"
layers = 26
release = "gemma-scope-2b-pt-res-canonical"
model = sae_lens.HookedSAETransformer.from_pretrained_no_processing(
    model_name, dtype=torch.bfloat16
)
saes = []
for layer in tqdm(range(layers)):
    if release == "gemma-scope-2b-pt-res-canonical":
        sae_id = f"layer_{layer}/width_16k/canonical"
    elif release == "llama_scope_lxr_8x":
        sae_id = f"l{layer}r_8x"
    elif release == "pythia-70m-deduped-res-sm":
        sae_id = f"blocks.{layer}.hook_resid_post"
    sae = sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0].to(torch.bfloat16)
    saes.append(sae)
    
layers = 26
top5_indices_wiki = [[] for _ in range(layers)]
top5_indices_math = [[] for _ in range(layers)]
last_5_tokens = -5
top5 = 5
top5_indices_code = [[] for _ in range(layers)]

for idx in range(1):
    # dataset, ratio, text = load_dataset_by_name("Math")
    # dataset = dataset.select(range(int(len(dataset) * ratio)))
    # for idy, example in enumerate(tqdm(dataset)):
    #     tokens = model.to_tokens([example[text]], prepend_bos=True)
    #     _, cache = model.run_with_cache_with_saes(
    #         tokens, saes=saes, use_error_term=True
    #     )
    #     for layer in range(layers):
    #         act_prompt = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
    #         top5_res = torch.topk(cache[act_prompt][:, last_5_tokens:, :], top5, dim=-1)
    #         top5_indices = top5_res.indices
    #         top5_indices = top5_indices.cpu().numpy()
    #         top5_indices_math[layer].append(top5_indices.flatten())
    # torch.save(
    #     {
    #         "top5_indices_math": top5_indices_math,
    #     },
    #     "top5_acts_gemma2_math.pt",
    # )
    dataset, ratio, text = load_dataset_by_name("wikitext")
    dataset = dataset.select(range(int(len(dataset) * 0.5)))
    with torch.no_grad():
        model.eval()
        for idy, example in enumerate(tqdm(dataset)):
            tokens = model.to_tokens([example[text]], prepend_bos=True)[:200]
            _, cache = model.run_with_cache_with_saes(
                tokens, saes=saes, use_error_term=True
            )
            model.reset_hooks()
            for layer in range(layers):
                act_prompt = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
                top5_res = torch.topk(cache[act_prompt][:, last_5_tokens:, :], top5, dim=-1)
                top5_indices = top5_res.indices.to(torch.float32)
                top5_indices = top5_indices.cpu().numpy()
                top5_indices_wiki[layer].append(top5_indices.flatten())
            torch.cuda.empty_cache()
        torch.save(
            {
                "top5_indices_wiki": top5_indices_wiki,
            },
            "top5_acts_gemma2_wiki.pt",
        )
    # dataset, ratio, text = load_dataset_by_name("sql-create-context")
    # dataset = dataset.select(range(int(len(dataset) * ratio)))
    # for idy, example in enumerate(tqdm(dataset)):
    #     tokens = model.to_tokens([example[text]], prepend_bos=True)
    #     _, cache = model.run_with_cache_with_saes(
    #         tokens, saes=saes, use_error_term=True
    #     )
    #     for layer in range(layers):
    #         act_prompt = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
    #         top5_res = torch.topk(cache[act_prompt][:, last_5_tokens:, :], top5, dim=-1)
    #         top5_indices = top5_res.indices
    #         top5_indices = top5_indices.to(torch.float32).cpu().numpy()
    #         top5_indices_code[layer].append(top5_indices.flatten())
    # torch.save(
    #     {
    #         "top5_indices_code": top5_indices_code,
    #     },
    #     "top5_acts_gemma2_code.pt",
    # )