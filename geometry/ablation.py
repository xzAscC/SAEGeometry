import torch
from tqdm import tqdm
from typing import List, Tuple
import sae_lens
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from geometry.data import fetch_sae_and_model, load_acts_from_pretrained, load_dataset_by_name
torch.set_grad_enabled(False)


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

    top_index_mc = np.setdiff1d(top_index_mc, top_index)
    top_index_mw = np.setdiff1d(top_index_mw, top_index)
    top_index_cw = np.setdiff1d(top_index_cw, top_index)
    
    top_index_wiki = np.setdiff1d(
        top_index_wiki.cpu().numpy(), np.union1d(top_index_cw, top_index_mw)
    )
    top_index_math = np.setdiff1d(
        top_index_math.cpu().numpy(), np.union1d(top_index_mc, top_index_mw)
    )
    top_index_code = np.setdiff1d(
        top_index_code.cpu().numpy(), np.union1d(top_index_mc, top_index_cw)
    )

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

def ablating(
    sae_name: str,
    layer: int,
    top_num: int,
):
    layers = 26
    math, code, wiki = [[] for _ in range(layers)], [[] for _ in range(layers)], [[] for _ in range(layers)]
    acts = load_acts_from_pretrained()
    for i in range(layers):
        _, _, _, code_index, math_index, wiki_index, _, _, _, _ = get_top_index(acts, 300, i)
        math[i] = math_index
        code[i] = code_index
        wiki[i] = wiki_index
    dataset_name = "wikitext"
    dataset, ratio, text = load_dataset_by_name(dataset_name)
    model_name = "gemma-2-2b"
    model = sae_lens.HookedSAETransformer.from_pretrained(model_name, dtype=torch.bfloat16)
    abl_times = 10
    layers = 26
    loss_softmax = [[0 for _ in range(abl_times)] for _ in range(layers)]
    loss_l1 = [[0 for _ in range(abl_times)] for _ in range(layers)]
    torch.set_grad_enabled(False)
    use_error_term = True
    ds_ratio = 1e-3
    dataset_length = int(len(dataset) * ds_ratio)
    pbar = tqdm(total=layers * abl_times * dataset_length)
    for layer in range(layers):
        release = "gemma-scope-2b-pt-res-canonical"
        sae_id = f"layer_{layer}/width_16k/canonical"
        sae1 = sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0]
        for idy in range(abl_times):
            doc_len = 0
            sae2 = sae_lens.SAE.from_pretrained(release, sae_id, device="cuda")[0]
            # random_indices = torch.randint(0, 32577, (900,))
            # list(map(lambda idx: sae2.W_dec[idx, :].zero_(), random_indices))
            high_freq_ind = wiki[layer][idy*10 : (idy + 1)*10]
            list(map(lambda idx: sae2.W_dec[idx, :].zero_(), high_freq_ind))
            
            for idx in range(dataset_length):
                example = dataset[idx]
                tokens = model.to_tokens([example[text]], prepend_bos=True)
                
                loss1, cache1 = model.run_with_cache_with_saes(
                    tokens, saes=sae1, use_error_term=use_error_term
                )
                model.reset_saes()

                loss2, cache2 = model.run_with_cache_with_saes(
                    tokens, saes=sae2, use_error_term=use_error_term
                )
                model.reset_saes()
                
                loss_softmax[layer][idy] += ((loss1.softmax(-1) - loss2.softmax(-1)) ** 2).sum().item()
                loss_l1[layer][idy] += (loss1.softmax(-1) - loss2.softmax(-1)).sum().item()
                pbar.update(1)
    pbar.close()
    
    mean_loss_softmax = [np.max(loss_softmax[layer]) for layer in range(layers)]

    sns.lineplot(x=range(layers), y=mean_loss_softmax, label='Mean Loss Softmax')
    plt.xlabel('Layer')
    plt.ylabel('Mean Loss Softmax')
    plt.title('Mean Loss Softmax for All Layers')
    plt.legend()
    plt.show()
    # mean_loss_softmax = [np.mean(loss_softmax[layer]) for layer in range(layers)]

    # sns.lineplot(x=range(layers), y=mean_loss_softmax, label='Mean Loss Softmax')
    # plt.xlabel('Layer')
    # plt.ylabel('Mean Loss Softmax')
    # plt.title('Mean Loss Softmax for All Layers')
    # plt.legend()
    # plt.show()
    return loss_softmax, loss_l1