try:
    from agcn.graph import tools
except ModuleNotFoundError:
    from graph import tools

# DM: note: I have not tried this specific graph, I inferred the links from dance revolution's plotting code. We first
# need to check we have sensible results on the basic tasks

# Edge format: (origin, neighbor)
num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward = [[17, 15], [15,  0], [0, 16], [16, 18],                      # head
          [0,  1], [1,  8],                                           # body
          [1,  2], [2,  3], [3,  4],                                  # right arm
          [1,  5], [5,  6], [6,  7],                                  # left arm
          [8,  9], [9, 10], [10, 11], [11, 24], [11, 22], [22, 23],   # right leg
          [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]]  # left leg

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class DanceRevolutionGraph:
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
