import sys

import yaml
import os

import jax
import jax.numpy as jnp
import optax
from flax import serialization
import flax.linen as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb

import numpy as np

from encoder import ResNet50


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def make_update_fn(*, encoder_apply_fn, classifier_apply_fn, optimizer):
    def update_fn(params, resnet_params, resnet_batch_stats, opt_state, inputs, labels):
        features = encoder_apply_fn(
            {
                "params": resnet_params,
                "batch_stats": resnet_batch_stats,
            },
            inputs,
            train=False,
        )

        features = jax.lax.stop_gradient(features)

        def loss_fn(params):
            logits = classifier_apply_fn(params, features)

            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

            return loss, logits

        (loss, logits), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)

        loss = jax.lax.pmean(loss, axis_name='batch')
        grad = jax.lax.pmean(grad, axis_name='batch')

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, logits

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0, 3))


def make_predict_fn(*, encoder_apply_fn, classifier_apply_fn):
    def predict_fn(params, resnet_params, resnet_batch_stats, inputs):
        features = encoder_apply_fn(
            {
                "params": resnet_params,
                "batch_stats": resnet_batch_stats,
            },
            inputs,
            train=False,
        )
        return classifier_apply_fn(params, features)
    return jax.pmap(predict_fn, axis_name='batch', donate_argnums=())


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

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))  # Convert [C, H, W] to [H, W, C]
    ])

    if dataset_config['dataset'] == 'imagenet':
        train_dataset = ImageFolder(
            root=dataset_config['train_data_path'],
            transform=train_transform,
        )

        val_dataset = ImageFolder(
            root=dataset_config['val_data_path'],
            transform=val_transform,
        )
    else:
        raise 'There is no such dataset'

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=True,
        num_workers=dataset_config['num_workers'],
        pin_memory=False,
        drop_last=True,
        prefetch_factor=dataset_config['prefetch_factor'],
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )

    encoder = ResNet50()

    classifier = nn.Dense(
        simsiam_config['num_classes'],
        kernel_init=nn.initializers.truncated_normal(stddev=0.01),
        bias_init=nn.initializers.zeros,
    )

    epochs = simsiam_config['linear_epochs']

    run = wandb.init(
        project=wandb_config['project'],
        name="LinearEval",
        reinit=True,
        config=config
    )

    pretrained_checkpoint_path = simsiam_config['checkpoint_path']

    pretrained_variables = load_checkpoint(pretrained_checkpoint_path, None)
    linear_eval_checkpoint_path = simsiam_config['linear_eval_checkpoint_path']

    resnet_params = pretrained_variables["params"]["encoder"]["ResNet_50_0"]
    resnet_batch_stats = pretrained_variables["batch_stats"]["encoder"]["ResNet_50_0"]

    key = jax.random.PRNGKey(seed)
    params = classifier.init(key, jnp.ones((2, 2048)))

    init_lr = simsiam_config['optim_params']['learning_rate'] * dataset_config['batch_size'] / 256

    optimizer = optax.sgd(learning_rate=init_lr, momentum=0.9, nesterov=False)

    opt_state = optimizer.init(params)

    replicate = lambda tree: jax.device_put_replicated(tree, jax.local_devices())
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    update_fn = make_update_fn(
        encoder_apply_fn=encoder.apply,
        classifier_apply_fn=classifier.apply,
        optimizer=optimizer,
    )

    predict_fn = make_predict_fn(
        encoder_apply_fn=encoder.apply,
        classifier_apply_fn=classifier.apply,
    )

    params_repl = replicate(params)
    resnet_params_repl = replicate(resnet_params)
    resnet_batch_stats_repl = replicate(resnet_batch_stats)
    opt_state_repl = replicate(opt_state)

    state_template = {
        "params": unreplicate(params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "epoch": 0,
    }

    del params
    del resnet_params
    del resnet_batch_stats
    del opt_state

    loaded_state = load_checkpoint(linear_eval_checkpoint_path, state_template)
    start_epoch = 0

    if loaded_state:
        params_repl = replicate(loaded_state['params'])
        opt_state_repl = replicate(loaded_state['opt_state'])
        start_epoch = loaded_state['epoch'] + 1

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (jax.local_device_count(), n // jax.local_device_count(), *s))

    def unshard(x):
        ndev, bs, *s = x.shape
        return jnp.reshape(x, (ndev * bs, *s))


    for epoch in range(start_epoch, epochs):
        for step, (images, labels) in enumerate(train_loader):
            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)
            labels = jax.tree_util.tree_map(lambda x: shard(np.array(x)), labels)

            (
                params_repl,
                opt_state_repl,
                loss,
                logits,
            ) = update_fn(
                params_repl,
                resnet_params_repl,
                resnet_batch_stats_repl,
                opt_state_repl,
                images,
                labels,
            )

            loss = unreplicate(loss)

            logits = unshard(logits)  # [N, C]
            labels = unshard(labels)  # [N]

            predictions = jnp.argmax(logits, axis=-1)  # [N]
            accuracy = jnp.mean(predictions == labels)

            print("Epoch: {} Step: {} Loss: {:.4f}".format(epoch, step, float(loss)))

            run.log({
                "loss": loss,
                "train_accuracy": accuracy,
                "epoch": epoch,
            })

        val_acc = []
        for images, labels in val_loader:
            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)
            labels_np = labels.numpy()

            logits_repl = predict_fn(params_repl, resnet_params_repl, resnet_batch_stats_repl, images)
            logits = unshard(logits_repl)

            predictions = logits.argmax(axis=-1)
            accuracy = jnp.mean(predictions == labels_np)
            val_acc.append(accuracy)

        print("Epoch {} Val Acc: {:.4f}".format(epoch + 1, jnp.mean(val_acc)))

        save_checkpoint(linear_eval_checkpoint_path, {
            "params": unreplicate(params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "epoch": epoch + 1,
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
