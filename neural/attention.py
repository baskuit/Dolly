from _torch.nn.activation import MultiheadAttention
import torch

from typing import List


def prod(a, b):
    for i in range(a):
        for j in range(b):
            yield ((i, j))


# class GroupedMultiheadCrossAttention(torch.nn.Module):
#     def __init__(
#         self,
#         col_group_features=(2**7, 2**6),
#         row_group_features=(2**4, 2**6),
#         inner_feature_dict=None,
#         outer_dim=2**6,
#         num_heads=1,
#         dropout=0.0,
#         bias=True,
#         batch_first=True,
#     ) -> None:

#         super().__init__()

#         self.m_groups = len(row_group_features)
#         self.n_groups = len(col_group_features)

#         if inner_feature_dict is None:
#             inner_feature_dict = {}
#             for a in range(self.m_groups):
#                 f = row_group_features[a]
#                 for b in range(self.n_groups):
#                     g = col_group_features[b]
#                     inner_feature_dict[(a, b)] = min(f, g)

#         self.cross_attn = torch.nn.ParameterDict(
#             {
#                 f"({a}, {b})": MultiheadAttention(
#                     embed_dim=row_group_features[a],
#                     inner_dim=inner_feature_dict[(a, b)],
#                     outer_dim=inner_feature_dict[(a, b)],
#                     kdim=col_group_features[b],
#                     vdim=col_group_features[b],
#                     output_dim=outer_dim,
#                     num_heads=num_heads,
#                     dropout=dropout,
#                     bias=bias,
#                     batch_first=batch_first,
#                 )
#                 for a, b in prod(self.m_groups, self.n_groups)
#             }
#         )

#     def forward(self, words: List[torch.Tensor], cross_words: List[torch.Tensor]):
#         words_ = []
#         for a in range(self.m_groups):
#             n, batch, _ = words[a].shape

#             group_outputs = torch.empty(
#                 size=(n, batch, 256, self.n_groups), device=words[a].device
#             )
#             group_weights = torch.empty(
#                 size=(n, batch, 1, self.n_groups), device=words[a].device
#             )

#             for b in range(self.n_groups):
#                 layer = self.cross_attn[f"({a}, {b})"]
#                 attn_outputs, logits = layer(words[a], cross_words[b], cross_words[b])
#                 group_outputs[..., b] = attn_outputs
#                 group_weights[..., 0, b] = torch.sum(torch.exp(logits), dim=-1)
#             group_weights = torch.nn.functional.normalize(group_weights, dim=-1, p=1)
#             group_outputs *= group_weights
#             word_ = torch.sum(group_outputs, dim=-1)
#             words_.append(word_)
#         return words_

# class GroupedTransformerEncoderLayer(torch.nn.Module):
#     def __init__(
#         self,
#         col_group_features=(2**7, 2**6),
#         row_group_features=(2**4, 2**6),
#         inner_feature_dict=None,
#         num_heads=1,
#         feed_forward_features=(2**8, 2**6, 2**7, 2**6),
#         dropout=0.0,
#         bias=True,
#         batch_first=True,
#     ) -> None:
#         super().__init__()

#         self.n_groups = len(group_features)

#         self.grouped_attention = GroupedMultiheadSelfAttention(
#             group_features=group_features,
#             inner_feature_dict=inner_feature_dict,
#             num_heads=num_heads,
#             dropout=dropout,
#             bias=bias,
#             batch_first=batch_first,
#         )
#         self.inner_linear_maps = torch.nn.ParameterList(
#             [
#                 torch.nn.Linear(group_features[a], feed_forward_features[a])
#                 for a in range(self.n_groups)
#             ]
#         )
#         self.outer_linear_maps = torch.nn.ParameterList(
#             [
#                 torch.nn.Linear(feed_forward_features[a], group_features[a])
#                 for a in range(self.n_groups)
#             ]
#         )
#         self.inner_layer_norms = torch.nn.ParameterList(
#             [
#                 torch.nn.LayerNorm(group_features[a], eps=10**-7)
#                 for a in range(self.n_groups)
#             ]
#         )
#         self.outer_layer_norms = torch.nn.ParameterList(
#             [
#                 torch.nn.LayerNorm(group_features[a], eps=10**-7)
#                 for a in range(self.n_groups)
#             ]
#         )

#     def forward(self, words: List[torch.Tensor]):
#         words_ = self.grouped_attention(words)
#         words = [
#             norm(word + word_)
#             for word, word_, norm in zip(words, words_, self.inner_layer_norms)
#         ]
#         words_ = [
#             outer_layer(torch.relu(inner_layer(word)))
#             for word, inner_layer, outer_layer in zip(
#                 words, self.inner_linear_maps, self.outer_linear_maps
#             )
#         ]
#         words = [
#             norm(word + word_)
#             for word, word_, norm in zip(words, words_, self.outer_layer_norms)
#         ]
#         return words


class GroupedMultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        group_features=(512, 256, 512, 256),
        inner_feature_dict=None,
        output_features=None,
        num_heads=1,
        dropout=0.0,
        bias=True,
        batch_first=True,
    ) -> None:

        super().__init__()

        self.n_groups = len(group_features)

        if inner_feature_dict is None:
            inner_feature_dict = {}
            for a in range(self.n_groups):
                f = group_features[a]
                for b in range(self.n_groups):
                    g = group_features[b]
                    inner_feature_dict[(a, b)] = min(f, g) // 2
        if output_features is None:
            output_features = group_features
        self.output_features = output_features

        self.class_attn_modules = torch.nn.ParameterDict(
            {
                f"({a}, {b})": MultiheadAttention(
                    embed_dim=group_features[a],
                    inner_dim=inner_feature_dict[(a, b)],
                    outer_dim=inner_feature_dict[(a, b)],
                    kdim=group_features[b],
                    vdim=group_features[b],
                    output_dim=output_features[a],
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                    batch_first=batch_first,
                )
                for a, b in prod(self.n_groups, self.n_groups)
            }
        )

    def forward(self, words: List[torch.Tensor]):
        outputs = []
        for a in range(self.n_groups):
            n, b, _ = words[a].shape
            d = self.output_features[a]
            group_outputs = torch.empty(
                size=(n, b, d, self.n_groups), device=words[a].device
            )
            group_weights = torch.empty(
                size=(n, b, 1, self.n_groups), device=words[a].device
            )

            for b in range(self.n_groups):
                layer = self.class_attn_modules[f"({a}, {b})"]
                attn_outputs, logits = layer(words[a], words[b], words[b])
                group_outputs[..., b] = attn_outputs
                group_weights[..., 0, b] = torch.sum(torch.exp(logits), dim=-1)
            group_weights = torch.nn.functional.normalize(group_weights, dim=-1, p=1)

            group_outputs *= group_weights
            outputs.append(torch.sum(group_outputs, dim=-1))
        return outputs


class GroupedTransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        group_features=(512, 256, 512, 256),
        inner_feature_dict=None,
        num_heads=1,
        feed_forward_features=(1024, 512, 1024, 512),
        dropout=0.0,
        bias=True,
        batch_first=True,
    ) -> None:
        super().__init__()

        self.n_groups = len(group_features)

        self.grouped_attention = GroupedMultiheadSelfAttention(
            group_features=group_features,
            inner_feature_dict=inner_feature_dict,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        self.inner_linear_maps = torch.nn.ParameterList(
            [
                torch.nn.Linear(group_features[a], feed_forward_features[a])
                for a in range(self.n_groups)
            ]
        )
        self.outer_linear_maps = torch.nn.ParameterList(
            [
                torch.nn.Linear(feed_forward_features[a], group_features[a])
                for a in range(self.n_groups)
            ]
        )
        self.inner_layer_norms = torch.nn.ParameterList(
            [
                torch.nn.LayerNorm(group_features[a], eps=10**-7)
                for a in range(self.n_groups)
            ]
        )
        self.outer_layer_norms = torch.nn.ParameterList(
            [
                torch.nn.LayerNorm(group_features[a], eps=10**-7)
                for a in range(self.n_groups)
            ]
        )

    def forward(self, words: List[torch.Tensor]):
        words_ = self.grouped_attention(words)
        words = [
            norm(word + word_)
            for word, word_, norm in zip(words, words_, self.inner_layer_norms)
        ]
        words_ = [
            outer_layer(torch.relu(inner_layer(word)))
            for word, inner_layer, outer_layer in zip(
                words, self.inner_linear_maps, self.outer_linear_maps
            )
        ]
        words = [
            norm(word + word_)
            for word, word_, norm in zip(words, words_, self.outer_layer_norms)
        ]
        return words


class GroupedTransformerEncoderBody(torch.nn.Module):
    def __init__(
        self,
        depth,
        group_features=(512, 256, 512, 256),
        inner_feature_dict=None,
        num_heads=1,
        feed_forward_features=(512, 256, 512, 256),
        dropout=0.0,
        bias=True,
        batch_first=True,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ParameterList(
            [
                GroupedTransformerEncoderLayer(
                    group_features=group_features,
                    inner_feature_dict=inner_feature_dict,
                    num_heads=num_heads,
                    feed_forward_features=feed_forward_features,
                    dropout=dropout,
                    bias=bias,
                    batch_first=batch_first,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, words):
        for layer in self.layers:
            words = layer(words)
        return words
