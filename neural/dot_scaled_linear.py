from _torch.nn.activation import DotScaledLinear
import torch

from typing import List


def prod(a, b):
    for i in range(a):
        for j in range(b):
            yield ((i, j))


class GroupedDotScaledLinear(torch.nn.Module):
    def __init__(
        self,
        row_dims=(2**10,),
        col_dims=(2**10,),
        out_dims=(2**10,),
        dot_dim_scale=2,  # this times min(col/row dim) is dot dim
    ) -> None:
        super().__init__()
        self.row_groups = len(row_dims)
        self.col_groups = len(col_dims)
        self.pairs = torch.nn.ParameterList(
            DotScaledLinear(
                row_dim=row_dims[i],
                col_dim=col_dims[j],
                dot_dim=int(dot_dim_scale * min(row_dims[i], col_dims[j])),
                output_dim=out_dims[i],
            )
            for i, j in prod(self.row_groups, self.col_groups)
        )

    def forward(self, row_groups: List[torch.Tensor], col_groups: List[torch.Tensor]):
        output = []
        for i in range(self.row_groups):
            row_words = row_groups[i]
            words = torch.stack(
                [
                    (self.pairs[i * self.col_groups + j])(row_words, col_words)[0]
                    for j, col_words in enumerate(col_groups)
                ],
                dim=-1,
            ).sum(-1)
            output.append(words)
        return output


class BatchNormWord(torch.nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super(BatchNormWord, self).__init__(*args, **kwargs)

    def apply(self, input):
        return self(input.swapaxes(-1, -2)).swapaxes(-1, -2)


class DSLResBlock(torch.nn.Module):
    def __init__(
        self,
        row_dims=(2**10,),
        col_dims=(2**10,),
        group_lengths=(
            1,
            5,
            1,
            5,
        ),
        dot_dim_scale=2,
    ) -> None:
        super().__init__()

        self.gdsl_1 = GroupedDotScaledLinear(
            row_dims=row_dims,
            col_dims=col_dims,
            out_dims=row_dims,
            dot_dim_scale=dot_dim_scale,
        )
        self.gdsl_2 = GroupedDotScaledLinear(
            row_dims=row_dims,
            col_dims=col_dims,
            out_dims=row_dims,
            dot_dim_scale=dot_dim_scale,
        )
        self.norms_1: List[torch.nn.LayerNorm] = torch.nn.ParameterList(
            torch.nn.LayerNorm((group_length, row_dim))
            for row_dim, group_length in zip(row_dims, group_lengths)
        )
        self.norms_2: List[torch.nn.LayerNorm] = torch.nn.ParameterList(
            torch.nn.LayerNorm((group_length, row_dim))
            for row_dim, group_length in zip(row_dims, group_lengths)
        )

    def forward(self, row_groups, col_groups):

        output = self.gdsl_1(row_groups, col_groups)
        output = [norm(torch.relu(words)) for words, norm in zip(output, self.norms_1)]
        output = self.gdsl_2(output, col_groups)
        output = [
            row_words + norm(words)
            for row_words, words, norm in zip(row_groups, output, self.norms_2)
        ]
        return output


class SelfDSLTower(torch.nn.Module):
    def __init__(
        self,
        depth=1,
        row_dims=(2**10,),
        dot_dim_scale=2,
        group_lengths=(
            1,
            5,
            1,
            5,
        ),
    ):
        super().__init__()
        self.tower = torch.nn.ParameterList(
            [
                DSLResBlock(
                    row_dims=row_dims,
                    col_dims=row_dims,
                    dot_dim_scale=dot_dim_scale,
                    group_lengths=group_lengths,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, groups):
        for layer in self.tower:
            groups = layer(groups, groups)
        return groups


class CrossDSLTower(torch.nn.Module):
    def __init__(
        self,
        depth=1,
        row_dims=(2**10,),
        col_dims=(2**10,),
        dot_dim_scale=2,
        group_lengths=(4, 5),
    ):
        super().__init__()
        self.tower = torch.nn.ParameterList(
            [
                DSLResBlock(
                    row_dims=row_dims,
                    col_dims=col_dims,
                    dot_dim_scale=dot_dim_scale,
                    group_lengths=group_lengths,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, row_groups, col_groups):
        for layer in self.tower:
            row_groups = layer(row_groups, col_groups)
        return row_groups
