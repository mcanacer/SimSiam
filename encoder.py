import jax
import jax.numpy as jnp

from flax import linen as nn


class MLP(nn.Module):
    in_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(self.in_dim, use_bias=False)(x)
        x = nn.BatchNorm(momentum=0.9, axis_name='batch')(x, use_running_average=not train)
        x = nn.relu(x)

        x = nn.Dense(self.in_dim, use_bias=False)(x)
        x = nn.BatchNorm(momentum=0.9, axis_name='batch')(x, use_running_average=not train)
        x = nn.relu(x)

        x = nn.Dense(self.out_dim, use_bias=False)(x)
        x = nn.BatchNorm(
            momentum=0.9,
            use_scale=False,
            use_bias=False,
            axis_name='batch'
        )(x, use_running_average=not train)

        return x


class Bottleneck(nn.Module):
    channels: int
    strides: int = 1
    zero_init_residual: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        residual = x

        y = nn.Conv(self.channels, (1, 1), strides=1, use_bias=False)(x)
        y = nn.BatchNorm(momentum=0.9, axis_name='batch')(y, use_running_average=not train)
        y = nn.relu(y)

        y = nn.Conv(self.channels, (3, 3), strides=self.strides, padding="SAME", use_bias=False)(y)
        y = nn.BatchNorm(momentum=0.9, axis_name='batch')(y, use_running_average=not train)
        y = nn.relu(y)

        y = nn.Conv(self.channels * 4, (1, 1), strides=1, use_bias=False)(y)

        # zero-init residual
        y = nn.BatchNorm(
            scale_init=(nn.initializers.zeros if self.zero_init_residual else nn.initializers.ones),
            axis_name = 'batch'
        )(y, use_running_average=not train)

        if residual.shape != y.shape:
            residual = nn.Conv(self.channels * 4, (1, 1),
                               strides=self.strides, use_bias=False)(residual)
            residual = nn.BatchNorm(momentum=0.9, axis_name='batch')(residual, use_running_average=not train)

        return nn.relu(residual + y)


class ResNet50(nn.Module):
    zero_init_residual: bool = True

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(64, (7, 7), strides=2, padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(momentum=0.9, axis_name='batch')(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        def make_layer(x, channels, num_blocks, stride_first=1):
            for i in range(num_blocks):
                x = Bottleneck(
                    channels=channels,
                    strides=stride_first if i == 0 else 1,
                    zero_init_residual=self.zero_init_residual
                )(x, train=train)
            return x

        x = make_layer(x, 64, 3)
        x = make_layer(x, 128, 4, stride_first=2)
        x = make_layer(x, 256, 6, stride_first=2)
        x = make_layer(x, 512, 3, stride_first=2)

        x = jnp.mean(x, axis=(1, 2))  # [N, 2048]

        return x


class Encoder(nn.Module):
    dim: int
    out_dim: int
    zero_init_residual: bool = True

    @nn.compact
    def __call__(self, x, train=True):
        x = ResNet50(self.zero_init_residual)(x, train=train)
        x = MLP(self.dim, self.out_dim)(x, train=train)
        return x
