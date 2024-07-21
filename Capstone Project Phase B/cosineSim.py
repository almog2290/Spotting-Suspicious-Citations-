import torch

def padVectors (vector1 , vector2):
    """
    This function pads the shorter of two input vectors with zeros to make them of equal length.

    Parameters:
    vector1 (torch.Tensor): The first input vector.
    vector2 (torch.Tensor): The second input vector.

    Returns:
    vector1 (torch.Tensor): The first vector, possibly padded with zeros.
    vector2 (torch.Tensor): The second vector, possibly padded with zeros.
    """

    if vector1.shape[1] > vector2.shape[1]:
        pad = torch.zeros(1,vector1.shape[1]-vector2.shape[1])
        vector2 = torch.cat((vector2,pad),1)
    elif vector1.shape[1] < vector2.shape[1]:                           
        pad = torch.zeros(1,vector2.shape[1]-vector1.shape[1])
        vector1 = torch.cat((vector1,pad),1)
    return vector1,vector2

def calc_cosine_similarity(vector1, vector2):
    """
    Calculates the cosine similarity between two vectors.

    Args:
        vector1 (torch.Tensor): The first vector.
        vector2 (torch.Tensor): The second vector.

    Returns:
        float: The cosine similarity score between the two vectors.
    """

    vecPad1, vecPad2 = padVectors(vector1, vector2)
    cosine_similarity = torch.nn.functional.cosine_similarity(vecPad1, vecPad2)
    similarity_score = cosine_similarity.item()
    return similarity_score


def createGmaeVectors (model , dm) :
    """
    This function generates game vectors by processing batches of data through a model. 
    The vectors are created by taking the mean of the model's output embeddings for each batch.

    Parameters:
    model (Model): The model to use for generating embeddings.
    dm (DataManager): The data manager, which provides the data loader for prediction.

    Returns:
    gmaeVectors (dict): A dictionary mapping node IDs to their corresponding game vectors.
    """
    
    gmaeVectors = {}
    predict_loader = dm.predict_dataloader()
    print('num of all nodes ->',len(predict_loader))

    for batched_data in predict_loader:
        batched_data = batched_data.to(model.device)
        output=model.generate_pretrain_embeddings_for_downstream_task2(batched_data)
        poolingVector = output[0].mean(dim=0, keepdim=True) # mean pooling
        gmaeVectors[batched_data.node_id[0].item()] = poolingVector

    return gmaeVectors


def calc_edges_restore (edge_dict , gmaeVectors ,threshold , dm):
    """
    This function calculates the accuracy of edge restoration based on cosine similarity scores.
    It iterates over each edge in the edge dictionary, calculates the cosine similarity between the game vectors of the nodes of the edge, and checks if the similarity score is above a certain threshold.
    If the edge was removed in this round and the similarity score is above the threshold, the edge is considered correctly restored.
    The function prints the maximum, minimum, and average similarity scores, as well as the number of correctly restored and manipulated edges, and the overall accuracy.

    Parameters:
    edge_dict (dict): A dictionary of edges.
    gmaeVectors (dict): A dictionary mapping node IDs to their corresponding game vectors.
    threshold (float): The threshold for considering an edge as correctly restored.
    dm (DataManager): The data manager, which provides information about the edges.

    Returns:
    acc (float): The accuracy of edge restoration.
    """

    if edge_dict is None:
        raise ValueError("edge_dict cannot be None")
    if gmaeVectors is None:
        raise ValueError("gmaeVectors cannot be None")
    if dm is None:
        raise ValueError("dm cannot be None")
    if threshold == 0:
        raise ValueError("threshold cannot be zero")    

    scoresVectors = []
    count_correct = 0
    count_manipulation = 0
    index = 0 

    #checking each edge by similarity of node_id1 and node_id2
    for key , edge in edge_dict.items():
        node_id1=edge[0]
        node_id2=edge[1]

        similarity_score = calc_cosine_similarity(gmaeVectors[node_id1],gmaeVectors[node_id2])
        scoresVectors.append(similarity_score)
       
        if dm.allUndirectedEdges[edge].isRemoveInThisRound == True :     
            if(similarity_score>threshold and threshold == 0.95) :
                dm.allUndirectedEdges[edge].numOfRecoveries_T95 = dm.allUndirectedEdges[edge].numOfRecoveries_T95 + 1
                count_correct = count_correct + 1
            elif (similarity_score>threshold and threshold == 0.9) :
                dm.allUndirectedEdges[edge].numOfRecoveries_T9 = dm.allUndirectedEdges[edge].numOfRecoveries_T9 + 1
                count_correct = count_correct + 1
            else:
                count_manipulation = count_manipulation + 1
        
        index = index + 1


    print('######### cosine similarity encoder -T={} #########'.format(threshold))
    print('max score ->',max(scoresVectors))
    print('min score ->',min(scoresVectors))
    print('avg score ->',sum(scoresVectors)/len(scoresVectors))
    print('######### edges accuracy -T={} ###################'.format(threshold))    
    print('size of correct edges ->',count_correct)
    print('size of manipulation edges ->',count_manipulation)
    acc=count_correct/(count_correct+count_manipulation)
    print('accuracy ->',acc)

    return acc





