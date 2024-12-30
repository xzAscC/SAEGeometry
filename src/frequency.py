import torch
import argparse
import sae_lens
import transformer_lens
import datasets
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
    return parser.parse_args()


def load_sae_from_saelens(sae_name: str, device: str) -> torch.nn.Module:
    sae_list = [] 
    match sae_name:
        case "pythia-70-res":
            layers = 6
            release = "pythia-70m-deduped-res-sm"
            sae_id = f"blocks.0.hook_resid_pre"
            model_name = "pythia-70m-deduped"
            sae_list.append(sae_lens.SAE.from_pretrained(release, sae_id)[0])
            for layer in range(layers):
                sae_id = f"blocks.{layer}.hook_resid_post"
                sae_list.append(sae_lens.SAE.from_pretrained(release, sae_id)[0])
            
            model = transformer_lens.HookedTransformer.from_pretrained(model_name)
            # TODO: add different datasets
            dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")['train']
        case "gpt2-small-res":
            pass
        case "get2-medium-res":
            pass
        case _:
            raise ValueError(f"Invalid SAE model name: {sae_name}")
        
    return sae_list, model, dataset


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
    logger.info(f"step 3: Geometry analysis")
    logger.info(f"step 3.1: vectors' cos sim in the same layer(max and min)")
    # TODO: pairwise in the same layer, too large to plot
    logger.info(f"step 3.2: cos sim with unembedding matrix")

    # TODO: here we do not care about the meaning, we only care about the cos sim and freq
    logger.info(f"step 4: Frequency analysis")
    logger.info(f"step 4.1: Plot avg frequency of the activation of the SAE")
    logger.info(f"step 4.2: cos sim with high freq, low freq and between them")
    logger.info(f"step 4.3: freq of the high cos sim, low cos sim")
    logger.info(
        f"step 4.4: freq of the high cos sim, low cos sim between the unembedding matrix"
    )

    logger.info(f"Then we can save the results and see the ablation study")
    logger.info(f"step 5: use different kinds of dataset to see the difference")
