import torch
import random
from Edge import Edge


def buildDictOfAllEdges (edge_index) :
    """
    This function builds a dictionary of all edges in an undirected graph.
    Each key in the dictionary is a tuple representing an edge (node1, node2), where node1 < node2.
    The value associated with each key is an Edge object, which contains information about the edge.
    If an edge (node2, node1) is encountered where node2 > node1, the index of this edge is added to the corresponding Edge object in the dictionary.

    Parameters:
    edge_index (torch.Tensor): A 2D tensor representing the edge index of the graph. Each column represents an edge.

    Returns:
    edge_dict (dict): A dictionary mapping edge tuples to Edge objects.
    """

    edge_dict = {}
    for i in range(edge_index.shape[1]):
        x, y = edge_index[0, i].item(), edge_index[1, i].item()
        if x < y and (x, y)  not in edge_dict:
            edge_dict[(x, y)] = Edge(i,0)
        else:
            # x > y and (y,x) in dict
            # add index of (x,y) to dict 
            edge_dict[(y, x)].edgeDirectTwo = i

    return edge_dict


def prepare_data(dm):
    """
    This function prepares the data for training and testing. It splits the nodes into training and testing sets, 
    initializes the training and testing edge dictionaries and masks, and iterates over all undirected edges in the data manager.
    For each edge, if both nodes are in the training set, the edge is added to the training edge dictionary and mask.
    If both nodes are in the testing set, the edge is added to the testing edge dictionary and mask, and the edge is marked as removed.
    The function also counts the number of directed edges, and the number of edges in the training and testing sets and masks.

    Parameters:
    dm (DataManager): The data manager, which provides the dataset and stores the training and testing node indices, edge dictionaries, and masks.

    Returns:
    None. The function modifies the data manager in-place.
    """

    if(dm.dataset.data.num_nodes == 0):
        raise ValueError("Number of nodes cannot be zero")

    total_nodes = dm.dataset.data.num_nodes
    num_train_nodes = int(total_nodes * 0.7)

    node_indices = list(range(total_nodes))
    random.shuffle(node_indices)

    dm.train_node_indices = torch.tensor(node_indices[:num_train_nodes])
    dm.test_node_indices = torch.tensor(node_indices[num_train_nodes:])

    #initlaize in each round
    dm.dictTrainEdges = {}
    dm.dictTestEdges = {}
    indexDirectedEdges = 0
    how_many_edges_train_mask = 0
    how_many_edges_test_mask = 0
    how_many_edges_train = 0
    how_many_edges_test = 0
    dm.train_mask_edges = torch.full(dm.dataset.data.edge_index[0, :].shape, False)
    dm.test_mask_edges = torch.full(dm.dataset.data.edge_index[0, :].shape, False)

    for key , edge in dm.allUndirectedEdges.items() :

        if(key[0] in dm.train_node_indices and key[1]  in dm.train_node_indices):
            dm.train_mask_edges[edge.edgeDirectOne] = True
            dm.train_mask_edges[edge.edgeDirectTwo] = True
            dm.dictTrainEdges[indexDirectedEdges] = (key[0],key[1])
            edge.isRemoveInThisRound = False
            how_many_edges_train = how_many_edges_train + 1
            how_many_edges_train_mask = how_many_edges_train_mask + 2
        elif(key[0] in dm.test_node_indices and key[1] in dm.test_node_indices):
            dm.test_mask_edges[edge.edgeDirectOne] = True
            dm.test_mask_edges[edge.edgeDirectTwo] = True
            dm.dictTestEdges[indexDirectedEdges] = (key[0],key[1])
            edge.numOfRemoved = edge.numOfRemoved + 1
            edge.isRemoveInThisRound = True
            how_many_edges_test = how_many_edges_test + 1
            how_many_edges_test_mask = how_many_edges_test_mask + 2
    
        indexDirectedEdges = indexDirectedEdges + 1


    print('how_many_edges_directed: ', indexDirectedEdges)
    print("train dict edges: ", how_many_edges_train)
    print("test dict edges: ", how_many_edges_test) 
    print("train mask edges: ", how_many_edges_train_mask)
    print("test mask edges: ", how_many_edges_test_mask)


######################  Unused method's right now - remove after refactor ####################
# # build a set of nodes that are in the testing set
# def buildSetOfTestingNode (model , dm) :

#     testing_node_set = set()
        
#     test_loader = dm.test_dataloader()
#     print('num of 40% nodes ->',len(test_loader))

#     for batched_data in test_loader:
#         batched_data = batched_data.to(model.device)
#         testing_node_set.add(batched_data.node_id[0].item())

#     return testing_node_set


# def add_random_edges_setByKnownTuple(dataset):
#     # load the manipulation edges from local file
#     manipulation_edges = loadFromLocalFile('manipulation_edges.pkl')
#     # Add the edge [x, y] and [y, x] to the edge_index
#     for (x,y) in manipulation_edges:
#         dataset.data.edge_index = torch.cat([dataset.data.edge_index, torch.tensor([[x, y], [y, x]])], dim=1)                

#     dataset.data.edge_index = sort_edge_index(dataset.data.edge_index)
#     return manipulation_edges

# def sort_edge_index(edge_index):
#     # Convert edge_index to numpy arrays
#     source_nodes = edge_index[0].numpy()
#     target_nodes = edge_index[1].numpy()

#     # Use numpy's lexsort function
#     sorted_indices = np.lexsort((target_nodes, source_nodes))

#     # Convert sorted_indices back to a torch tensor
#     sorted_indices = torch.from_numpy(sorted_indices)

#     # Use the sorted indices to sort edge_index
#     sorted_edge_index = edge_index.index_select(1, sorted_indices)

#     return sorted_edge_index

# def add_random_edges(dataset, num_edges_to_add):
#     # create a list of edges to add
#     manipulation_edges = []
#     while num_edges_to_add > 0:
        
#         # Generate two random numbers within the range of the node indices
#         x, y = torch.randint(2708, (2,))

#         # If x and y are equal, generate new x and y
#         if x == y:
#             continue

#         # Check if the edge already exists in the edge_index
#         if not ((dataset.data.edge_index[0] == x) & (dataset.data.edge_index[1] == y)).any():
#             manipulation_edges.append((x, y))
#             # Add the edge [x, y] and [y, x] to the edge_index
#             dataset.data.edge_index = torch.cat([dataset.data.edge_index, torch.tensor([[x, y], [y, x]])], dim=1)
#             num_edges_to_add =  num_edges_to_add - 1

#     # Sort the edge_index
#     dataset.data.edge_index = sort_edge_index(dataset.data.edge_index)
    
#     # Save the edges to local file
#     saveToLocalFile('manipulation_edges.pkl', manipulation_edges)
#     #manipulation_edges=loadFromLocalFile('manipulation_edges.pkl')

#     return manipulation_edges
##########################################################################################