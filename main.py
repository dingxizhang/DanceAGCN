import numpy as np
from graph.dance_revolution import DanceRevolutionGraph
from agcn.model.aagcn import Model


def new_aagcn(num_classes=3):
    graph = DanceRevolutionGraph(labeling_mode='spatial')
    model = Model(num_class=num_classes, num_point=graph.num_node, in_channels=2, graph=graph)
    return model


def run_batch(input_tensor, model):
    # DM: this is just an example to show how the data has to be passed to the model
    B, C, T, V, M = input_tensor.shape
    # input shape:
    # B: batch size,
    # C=2 (xy channels),
    # T: length of sequence (n. of frames),
    # V=25, number of nodes in the skeleton,
    # M=1 number of bodies

    output = model(input_tensor)
    # will return a classification output for each element in the batch, already averaged over time. There should be
    # a dimension with size 1 corresponding to the single body for which we have data

    return output

# TODO: create a PyTorch dataset object feeding DanceRevolution's skeleton data in the expected format.
#  Look at attached notebook to understand Dance Revolution dance format. Look also at dataset_holder.py to see how
#  data can be loaded first and then fed via a Dataset object. I'm including a stub object in dataset.py for your
#  reference




