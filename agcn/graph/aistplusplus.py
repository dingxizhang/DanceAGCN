try:
    from agcn.graph import tools
except ModuleNotFoundError:
    from graph import tools

# DM: note: I have not tried this specific graph, I inferred the links from dance revolution's plotting code. We first
# need to check we have sensible results on the basic tasks

# Edge format: (origin, neighbor)
num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14),
        (14, 16), (0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)]  # COCO-format bone constrains

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AISTplusplusGraph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print('')
