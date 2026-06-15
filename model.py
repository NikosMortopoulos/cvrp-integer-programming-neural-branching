import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer("shift", torch.zeros(n_units) if shift else None)
        self.register_buffer("scale", torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, (
            f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."
        )

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units

        delta = sample_avg - self.avg

        self.m2 = (
            self.var * self.count
            + sample_var * sample_count
            + delta ** 2 * self.count * sample_count / (self.count + sample_count)
        )

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        assert self.count > 0

        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size=64, edge_nfeats=1, dropout=0.05):
        super().__init__(aggr="add")

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )

        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(edge_nfeats, emb_size, bias=False)
        )

        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )

        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        output = self.post_conv_module(output)
        output = torch.cat([output, right_features], dim=-1)

        return self.output_module(output)

    def message(self, node_features_i, node_features_j, edge_features):
        return self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )


class BaseModel(torch.nn.Module):
    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True


class GNNPolicy(BaseModel):
    def __init__(
        self,
        var_nfeats=19,
        cons_nfeats=5,
        edge_nfeats=1,
        emb_size=64,
        n_rounds=2,
        dropout=0.05,
    ):
        super().__init__()

        self.emb_size = emb_size
        self.n_rounds = n_rounds

        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = torch.nn.ModuleList([
            BipartiteGraphConvolution(
                emb_size=emb_size,
                edge_nfeats=edge_nfeats,
                dropout=dropout,
            )
            for _ in range(n_rounds)
        ])

        self.conv_c_to_v = torch.nn.ModuleList([
            BipartiteGraphConvolution(
                emb_size=emb_size,
                edge_nfeats=edge_nfeats,
                dropout=dropout,
            )
            for _ in range(n_rounds)
        ])

        self.cons_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(emb_size)
            for _ in range(n_rounds)
        ])

        self.var_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(emb_size)
            for _ in range(n_rounds)
        ])

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        return_embeddings=False,
    ):
        reversed_edge_indices = torch.stack(
            [edge_indices[1], edge_indices[0]],
            dim=0,
        )

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        for r in range(self.n_rounds):
            old_cons = constraint_features
            old_vars = variable_features

            constraint_features = self.conv_v_to_c[r](
                variable_features,
                reversed_edge_indices,
                edge_features,
                constraint_features,
            )

            constraint_features = self.cons_norms[r](
                constraint_features + old_cons
            )

            variable_features = self.conv_c_to_v[r](
                constraint_features,
                edge_indices,
                edge_features,
                variable_features,
            )

            variable_features = self.var_norms[r](
                variable_features + old_vars
            )

        logits = self.output_module(variable_features).squeeze(-1)

        if return_embeddings:
            return logits, variable_features

        return logits

