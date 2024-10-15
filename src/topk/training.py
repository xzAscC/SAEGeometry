import os

import torch
import nnsight
import datasets
import wandb
import json
import queue

import multiprocessing as mp

from tqdm import tqdm

from buffer import ActivationBuffer
from dictionary import AutoEncoder, AutoEncoderTopK
from trainer import StandardTrainer, TrainerTopK

# TODO: link github
# TODO: why my black not function?


def new_wandb_process(config, log_queue, entity, project):
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            wandb.log(log)
        except queue.Empty:
            continue
    wandb.finish()


def log_stats(
    trainers,
    step: int,
    act: torch.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list=[],
):
    with torch.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act = z.clone()
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            if not transcoder:
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance explained
                total_variance = torch.var(act, dim=0).sum()
                residual_variance = torch.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            # log parameters from training
            log.update({f"{k}": v for k, v in losslog.items()})
            log[f"l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                log[f"{name}"] = value

            if log_queues:
                log_queues[i].put(log)


def trainSAE(
    data,
    trainer_configs,
    use_wandb=False,
    wandb_entity="",
    wandb_project="",
    steps=None,
    save_steps=None,
    save_dir=None,
    log_steps=None,
    activations_split_by_head=False,
    transcoder=False,
    run_cfg={},
):
    """
    Train SAEs using the given trainers
    """
    trainers = []
    for config in trainer_configs:
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    wandb_processes = []
    log_queues = []

    if use_wandb:
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            try:
                config["buffer"] = data.config
            except:
                pass
            with open(os.path.join(dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    for step, act in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            break

        # logging
        if log_steps is not None and step % log_steps == 0:
            log_stats(
                trainers, step, act, activations_split_by_head, transcoder, log_queues=log_queues
            )

        # saving
        if save_steps is not None and step % save_steps == 0:
            for dir, trainer in zip(save_dirs, trainers):
                if dir is not None:
                    if not os.path.exists(os.path.join(dir, "checkpoints")):
                        os.mkdir(os.path.join(dir, "checkpoints"))
                    torch.save(
                        trainer.ae.state_dict(),
                        os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                    )

        # training
        for trainer in trainers:
            trainer.update(step, act)

    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if save_dir is not None:
            torch.save(trainer.ae.state_dict(), os.path.join(save_dir, "ae.pt"))

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()


if __name__ == '__main__':

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_name = "roneneldan/TinyStories-1Layer-21M" # can be any Huggingface model, "roneneldan/TinyStories-1Layer-21M", "openai-community/gpt2"
    model = nnsight.LanguageModel(model_name, device=device)

    # see layer: model.config.num_layers
    #submodule = model.gpt_neox.layers[1].mlp # layer 1 MLP
    activation_dim = model.config.hidden_size # output dimension of the MLP
    enlarge_factor = 16
    dictionary_size = enlarge_factor * activation_dim

    if model_name == "roneneldan/TinyStories-1Layer-21M":
        # TODO: why tranfroms not get_neox
        submodule = model.transformer.h[0].mlp
        dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2"
        dataset = datasets.load_dataset(dataset_path)['train']

    dataset = iter(
    [
        "This is some example data",
        "In real life, for training a dictionary",
        "you would need much more data than this",
    ]
)
    buffer = ActivationBuffer(
        data=dataset,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim, # output dimension of the model component
        n_ctxs=3e4,  # you can set this higher or lower dependong on your available memory
        device=device,
    )  # buffer will yield batches of tensors of dimension = submodule's output dimension

    trainer_cfg = {
        "trainer": TrainerTopK, # StandardTrainer, TrainerTopK
        "dict_class": AutoEncoderTopK, # AutoEncoder, AutoEncoderTopK or AutoEncoderATopK
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": 1e-3,
        "device": device,
        "layer": 0,
        "lm_name": model_name,
        "submodule_name": submodule.__class__.__name__,
    }

    # train the sparse autoencoder (SAE)
    ae = trainSAE(
        data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
        trainer_configs=[trainer_cfg],
        use_wandb=True,
        save_dir="save_dir/topk1",
        log_steps=100,
        save_steps=1000,
    )