import torch
import torch.nn as nn

class Graph:
    """
    Simple graph definition for ST-GCN.

    Args:
        num_nodes (int): number of joints (V)
        edges (list of tuples): each (i, j) is an undirected edge between joints i and j
        strategy (str): adjacency splitting strategy, here we just use 'uniform'
    """
    def __init__(self, num_nodes, edges, strategy='uniform'):
        self.num_nodes = num_nodes
        self.edges = edges
        self.strategy = strategy

        # Build adjacency matrix A with shape (K, V, V).
        # Here we use K=1 (single adjacency), i.e. the 'uniform' strategy.
        self.A = self._build_adjacency()

    def _build_adjacency(self):
        V = self.num_nodes
        A = torch.zeros((1, V, V), dtype=torch.float32)  # K=1

        for i, j in self.edges:
            A[0, i, j] = 1.0
            A[0, j, i] = 1.0  # undirected

        # Add self-connections (optional but common)
        for v in range(V):
            A[0, v, v] = 1.0

        return A.numpy()

class ConvTemporalGraphical(nn.Module):
    """
    Spatial graph convolution used in ST-GCN.

    Args:
        in_channels (int): number of input feature channels
        out_channels (int): number of output feature channels
        K (int): number of adjacency partitions (A has shape (K, V, V))
    """

    def __init__(self, in_channels: int, out_channels: int, K: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        # 1x1 conv to create K groups of out_channels for each node
        # Input:  (N, C_in, T, V)
        # Output: (N, C_out * K, T, V)
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        x: (N, C_in, T, V); Tensor
        A: (K, V, V); Tensor

        Returns:
            x: (N, C_out, T, V)
            A: (K, V, V) (unchanged, just passed through)
        """
        N, C, T, V = x.size()
        K, _, _ = A.size()  # should equal self.K

        # 1) Channel mixing: 1x1 conv on (C, T, V)
        x = self.conv(x)  # -> (N, C_out * K, T, V)

        # 2) Split into K branches: (N, K, C_out, T, V)
        x = x.view(N, K, self.out_channels, T, V)

        # 3) Graph convolution:
        # for each partition k, x_k_out(v) = sum_u A[k, v, u] * x_k_in(u)
        # einsum: (N, K, C, T, V) x (K, V, V) -> (N, C, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)

        return x, A

class STGCNBlock(nn.Module):
    """
    One ST-GCN block: spatial graph conv + temporal conv + residual.
    Args:
        in_channels (int): C_in
        out_channels (int): C_out
        kernel_size (tuple): (temporal_kernel_size, K)
        stride (int): temporal stride (for downsampling in time)
        dropout (float): dropout rate
        residual (bool): whether to use residual connection
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1  # odd temporal kernel for same-padding

        temporal_kernel_size, K = kernel_size
        padding = ((temporal_kernel_size - 1) // 2, 0)

        # Spatial GCN over graph
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, K)

        # Temporal convolution (1D over time, implemented as Conv2d)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # Residual branch
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        """
        x: (N, C_in, T_in, V)
        A: (K, V, V)
        """
        # Residual path
        res = self.residual(x)  # (N, C_out, T_out, V)

        # Spatial GCN
        x, A = self.gcn(x, A)   # (N, C_out, T_in, V)

        # Temporal Conv + residual
        x = self.tcn(x) + res   # (N, C_out, T_out, V)

        return self.relu(x), A

class STGCNEncoder(nn.Module):
    """
    ST-GCN backbone as an encoder for sign language translation.

    Input:
        x: (N, C, T, V, M)

    Output:
        seq_feats: (T_out, N, D)
            - T_out: temporal length after downsampling
            - D: feature dimension (e.g. 256)
        plus optionally intermediate feature maps if you like.
    """

    def __init__(self,
                 in_channels,
                 graph_args,
                 edge_importance_weighting=True,
                 dropout=0.0):
        super().__init__()

        # 1) Load graph and adjacency
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)     # A: (K, V, V)

        # 2) Build ST-GCN block stack (similar to the original Model)
        spatial_kernel_size = A.size(0)  # K
        temporal_kernel_size = 9 ##TODO: consider changing for our purpose
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        V = A.size(1)
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        # For first block we disable dropout in residual path like original
        kwargs0 = dict(dropout=dropout)
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 128,  kernel_size, stride=1, residual=False, **kwargs0),
            STGCNBlock(128,         128,  kernel_size, stride=1, dropout=dropout),
            STGCNBlock(128,         128,  kernel_size, stride=1, dropout=dropout),
            STGCNBlock(128,         128,  kernel_size, stride=1, dropout=dropout),
            STGCNBlock(128,         128, kernel_size, stride=2, dropout=dropout),  # downsample T
            STGCNBlock(128,         128, kernel_size, stride=1, dropout=dropout),
            STGCNBlock(128,         128, kernel_size, stride=1, dropout=dropout),
            STGCNBlock(128,         256, kernel_size, stride=2, dropout=dropout),  # downsample T
            STGCNBlock(256,         256, kernel_size, stride=1, dropout=dropout),
            STGCNBlock(256,         256, kernel_size, stride=1, dropout=dropout),
        ))

        # 3) Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(self.A))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1.0] * len(self.st_gcn_networks)

        # Final feature dimension after last block
        self.output_dim = 256

    def forward(self, x):
        """
        x: (N, C, T, V, M)

        Returns:
            seq_feats: (T_out, N, D)  where D = self.output_dim
        """
        N, C, T, V, M = x.size()

        # ---- Data normalization (same as original code) ----
        # (N, C, T, V, M) -> (N*M, V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)                 # (N*M, C, T, V)

        # ---- ST-GCN blocks ----
        A = self.A  # (K, V, V)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, A = gcn(x, A * importance)  # x: (N*M, C', T', V)

        # ---- Reshape back to separate N and M ----
        _, C_out, T_out, V_out = x.size()
        x = x.view(N, M, C_out, T_out, V_out)  # (N, M, C_out, T_out, V)

        # ---- Pool over joints and persons ----
        # Average over joints V and persons M -> per-frame representation
        x = x.mean(dim=4).mean(dim=1)  # (N, C_out, T_out)

        # For Transformer: we usually want (T_out, N, D)
        seq_feats = x.permute(2, 0, 1).contiguous()  # (T_out, N, C_out)

        return seq_feats
