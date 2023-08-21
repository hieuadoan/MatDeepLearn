import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric.utils as pyg_utils
import networkx as nx
import matplotlib.pyplot as plt

class pygData(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_wSelfLoops.pt']

dataset = pygData('./mo2c_data/Oads_Mo2C_ini/')
idx = 1000
data = dataset[idx]
x = data.x
edge_index = data.edge_index
adj_mat = pyg_utils.to_dense_adj(edge_index)[0] 
row, col = edge_index
deg = pyg_utils.degree(row, x.size(0), dtype=x.dtype)
g = pyg_utils.to_networkx(data)
#pos = nx.spring_layout(g, seed=123456)
print(dataset)
print('Example graph data:', data.structure_id)
print('-------------------')
print(data)
print('---')
print('Tensor of atomic number:')
print(data.z)
print('---')
print('edge_index[0]',edge_index[0])
print('---')
print('edge_index[1]',edge_index[1])
print('---')
print('Node degree:')
print(deg, deg.shape)
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of edges: {data.num_edges}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
#print('---')
#print('SOAP features')
#print(data.extra_features_SOAP, data.extra_features_SOAP.shape)
nx.draw_networkx(g, with_labels=True, node_size=200, font_size=10)
#plt.tight_layout()
plt.show()
#plt.savefig('Graph_'+str(idx)+'.png', format='PNG')
