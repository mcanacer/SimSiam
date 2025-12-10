import sys
import yaml
import os

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb
from simsiam import SimSiam
from loader import GaussianBlur, TwoCropsTransform

import numpy as np


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def create_lr_schedule_fn(init_lr, total_steps, fix_pred_lr=True):
    backbone_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=init_lr,
        warmup_steps=int(0.1 * total_steps),
        decay_steps=total_steps,
        end_value=0.0
    )

    predictor_schedule = optax.constant_schedule(init_lr)

    def lr_fn(step, params):
        backbone_lr = backbone_schedule(step)
        predictor_lr = predictor_schedule(step)

        def get_lr_for_param(path, _):
            path_names = [str(p.key) if hasattr(p, 'key') else str(p) for p in path]
            full_path_str = '/'.join(path_names).lower()
            if fix_pred_lr and 'predictor' in full_path_str:
                return predictor_lr
            return backbone_lr

        return jax.tree_util.tree_map_with_path(get_lr_for_param, params)

    return lr_fn


def make_update_fn(*, apply_fn, lr_fn, momentum, weight_decay):
    def update_fn(params, batch_stats, opt_state, x1, x2, step):
        def loss_fn(params):
            (p1, p2, z1, z2), new_batch_stats = apply_fn(
                {
                    "params": params,
                    "batch_stats": batch_stats,
                },
                x1, x2,
                train=True,
                mutable=['batch_stats']
            )

            loss = -(
                    optax.losses.cosine_similarity(p1, z2, axis=1, epsilon=1e-8).mean() +
                    optax.losses.cosine_similarity(p2, z1, axis=1, epsilon=1e-8).mean()
            ) * 0.5

            return loss, new_batch_stats["batch_stats"]

        ((loss, new_batch_stats), grad) = jax.value_and_grad(loss_fn, has_aux=True)(params)

        loss = jax.lax.pmean(loss, axis_name='batch')
        grad = jax.lax.pmean(grad, axis_name='batch')

        lr_tree = lr_fn(step, params)

        new_opt_state = jax.tree_util.tree_map(
            lambda m, g: momentum * m + g,
            opt_state, grad
        )

        def apply_update(path, param, mom, lr):
            path_names = [str(p.key) if hasattr(p, 'key') else str(p) for p in path]
            is_no_decay = any(name in ('bias', 'scale') for name in path_names)

            if is_no_decay:
                wd = 0.0
            else:
                wd = weight_decay

            return param - lr * (mom + wd * param)

        new_params = jax.tree_util.tree_map_with_path(
            apply_update,
            params, new_opt_state, lr_tree
        )

        return new_params, new_batch_stats, new_opt_state, loss

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0, 1, 2))


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    simsiam_config = config['model']
    dataset_config = config['dataset_params']
    wandb_config = config['wandb']

    seed = simsiam_config['seed']

    augmentation = [
        transforms.RandomResizedCrop(dataset_config['img_size'], scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
    ]

    if dataset_config['dataset'] == 'imagenet':
        train_dataset = ImageFolder(
            root=dataset_config['data_path'],
            transform=TwoCropsTransform(transforms.Compose(augmentation)),
        )
    else:
        raise ValueError('There is no such dataset')

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=True,
        num_workers=dataset_config['num_workers'],
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=dataset_config['prefetch_factor'],
    )

    simsiam = SimSiam(**simsiam_config['params'])

    epochs = simsiam_config['epochs']

    run = wandb.init(
        project=wandb_config['project'],
        name=wandb_config['name'],
        reinit=True,
        config=config
    )

    checkpoint_path = simsiam_config['checkpoint_path']

    inputs, _ = next(iter(train_loader))
    x1, x2 = inputs

    key = jax.random.PRNGKey(seed)
    variables = simsiam.init(key, jnp.array(x1), jnp.array(x2))

    params, batch_stats = variables['params'], variables['batch_stats']

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * simsiam_config['epochs']

    init_lr = simsiam_config['optim_params']['learning_rate'] * dataset_config['batch_size'] / 256
    wd = simsiam_config['optim_params']['weight_decay']
    momentum = simsiam_config['optim_params']['momentum']

    fix_pred_lr = simsiam_config['optim_params'].get('fix_pred_lr', True)

    lr_fn = create_lr_schedule_fn(
        init_lr=init_lr,
        total_steps=total_steps,
        fix_pred_lr=fix_pred_lr
    )

    replicate = lambda tree: jax.device_put_replicated(tree, jax.local_devices())
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    opt_state = jax.tree_util.tree_map(jnp.zeros_like, params)

    update_fn = make_update_fn(
        apply_fn=simsiam.apply,
        lr_fn=lr_fn,
        momentum=momentum,
        weight_decay=wd,
    )

    params_repl = replicate(params)
    batch_stats_repl = replicate(batch_stats)
    opt_state_repl = replicate(opt_state)

    state_template = {
        "params": unreplicate(params_repl),
        "batch_stats": unreplicate(batch_stats_repl),
        "opt_state": unreplicate(opt_state_repl),
        "epoch": 0,
    }

    del params
    del batch_stats
    del opt_state

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    start_epoch = 0
    global_step = 0

    if loaded_state:
        params_repl = replicate(loaded_state['params'])
        batch_stats_repl = replicate(loaded_state['batch_stats'])
        opt_state_repl = replicate(loaded_state['opt_state'])
        start_epoch = loaded_state['epoch'] + 1
        global_step = start_epoch * steps_per_epoch

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (jax.local_device_count(), n // jax.local_device_count(), *s))

    def unshard(x):
        ndev, bs, *s = x.shape
        return jnp.reshape(x, (ndev * bs, *s))

    step_repl = replicate(global_step)

    for epoch in range(start_epoch, epochs):
        for step, (images, _) in enumerate(train_loader):
            x1, x2 = images[0], images[1]
            x1 = jax.tree_util.tree_map(lambda x: shard(np.array(x)), x1)
            x2 = jax.tree_util.tree_map(lambda x: shard(np.array(x)), x2)

            (
                params_repl,
                batch_stats_repl,
                opt_state_repl,
                loss
            ) = update_fn(
                params_repl,
                batch_stats_repl,
                opt_state_repl,
                x1,
                x2,
                step_repl
            )

            step_repl = step_repl + 1

            loss = unreplicate(loss)

            print("Epoch: {} Step: {} Loss: {:.4f}".format(epoch, step, float(loss)))

            run.log({
                "loss": loss,
                "epoch": epoch,
            })

        save_checkpoint(checkpoint_path, {
            "params": unreplicate(params_repl),
            "batch_stats": unreplicate(batch_stats_repl),
            "opt_state": unreplicate(opt_state_repl),
            "epoch": epoch + 1,
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
