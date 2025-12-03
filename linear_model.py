import jax

import flax.linen as nn

from encoder import ResNet50


class LinearModel(nn.Module):
    num_classes: int

    def setup(self):
        self.backbone = ResNet50()
        self.classifier = nn.Dense(
            self.num_classes,
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        x = self.backbone(x, train=False)
        x = jax.lax.stop_gradient(x)
        x = self.classifier(x)
        return x
