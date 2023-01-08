from neural.attention import (
    GroupedMultiheadSelfAttention,
    GroupedTransformerEncoderBody,
)
from neural.basket2 import BasketNet2
from _torch.nn.activation import DotScaledLinear
from neural.dot_scaled_linear import GroupedDotScaledLinear, SelfDSLTower

import torch

if __name__ == "__main__":

    # row_dims = (2**9,)
    # col_dims = (2**9,)

    # layer = GroupedDotScaledLinear(
    #     row_dims=row_dims,
    #     col_dims=col_dims,
    #     out_dims=row_dims,
    #     dot_dim_scale=2,
    # )

    # batch = 1
    # m = n = 5
    # row_words = [torch.rand((batch, m, row_dim)) for row_dim in row_dims]
    # col_words = [torch.rand((batch, m, col_dim)) for col_dim in col_dims]

    # output = layer.forward(row_words, col_words)
    # print(row_words[0])
    # print(output[0])
    net = SelfDSLTower(
        depth=4,
        row_dims=(2**7, 2**6, 2**7, 2**6),
        dot_dim_scale=2,
    )

    print(sum([_.numel() for _ in net.parameters() if _.requires_grad]))