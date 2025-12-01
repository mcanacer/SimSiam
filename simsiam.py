import jax
import flax.linen as nn

from encoder import Encoder


class Predictor(nn.Module):
    hidden_dim: int = 512
    out_dim: int = 2048

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(self.hidden_dim, use_bias=False)(x)
        x = nn.BatchNorm(momentum=0.9, axis_name='batch')(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class SimSiam(nn.Module):
    dim: int = 2048
    pred_dim: int = 512
    out_dim: int = 2048

    def setup(self):
        self.encoder = Encoder(dim=self.dim, out_dim=self.out_dim)
        self.predictor = Predictor(hidden_dim=self.pred_dim, out_dim=self.out_dim)

    def __call__(self, x1, x2, train=True):
        z1 = self.encoder(x1, train=train)  # [N, C]
        z2 = self.encoder(x2, train=train)  # [N, C]

        p1 = self.predictor(z1, train=train)  # [N, C]
        p2 = self.predictor(z2, train=train)  # [N, C]

        z1 = jax.lax.stop_gradient(z1)
        z2 = jax.lax.stop_gradient(z2)

        return p1, p2, z1, z2
