import torch
from torch_geometric.utils import to_dense_adj, degree, to_undirected
import pickle
import torch.nn.functional as F


def save_with_pickle(data, filename):
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(data, f)
    # print(f"Data saved as {filename}.pkl")


def load_with_pickle(filename):
    with open(f'{filename}.pkl', 'rb') as f:
        return pickle.load(f)


def get_adj(edge_index, num_nodes):
    '''PyG Data object to dense adjacency matrix'''
    device = edge_index.device
    adj = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.size(1)).to(device),
        size=(num_nodes, num_nodes),
    )

    # Calculate degree-normalized adjacency matrix
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    d_sp = torch.sparse_coo_tensor(
        torch.arange(num_nodes).repeat(2, 1).to(device),
        deg_inv_sqrt,
        size=(num_nodes, num_nodes),
    )
    norm_adj = d_sp @ adj
    return norm_adj


def maximum_spanning_tree(adj_matrix):
    """
    Find the maximum spanning tree of a fully-connected graph using Prim's algorithm.

    Args:
        adj_matrix: numpy array of shape (n, n) representing adjacency matrix

    Returns:
        List of tuples representing edges in the maximum spanning tree in format (u, v)
    """
    n = adj_matrix.shape[0]

    # Initialize
    visited = [False] * n
    max_edge = [(-float('inf'), -1, -1)] * n  # (weight, from, to)
    visited[0] = True

    # Initialize max_edge for all nodes connected to node 0
    for i in range(1, n):
        max_edge[i] = (adj_matrix[0][i], 0, i)

    edges = []

    # Build the MST
    for _ in range(n - 1):
        # Find the maximum weight edge from visited to unvisited nodes
        max_weight = -float('inf')
        next_node = -1
        from_node = -1

        for i in range(n):
            if not visited[i] and max_edge[i][0] > max_weight:
                max_weight = max_edge[i][0]
                next_node = i
                from_node = max_edge[i][1]

        if next_node == -1:
            break

        # Add the edge to MST
        edges.append((from_node, next_node))
        visited[next_node] = True

        # Update max_edge for unvisited nodes
        for i in range(n):
            if not visited[i] and adj_matrix[next_node][i] > max_edge[i][0]:
                max_edge[i] = (adj_matrix[next_node][i], next_node, i)

    return edges


def get_directed_ei(edge_index):
    return edge_index[:, edge_index[0] < edge_index[1]]


def pairwise_cos_sim(x1, x2):
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    return torch.mm(x1_norm, x2_norm.t())
