import pickle

def saveToLocalFile(FileName, dataToSave):
    """
    Save data to a local file using pickle.

    Args:
        FileName (str): The name of the file to save the data to.
        dataToSave: The data to be saved.

    Returns:
        None
    """

    with open(FileName, 'wb') as f:
        pickle.dump(dataToSave, f)

def loadFromLocalFile(FileName):
    """
    Load data from a local file.

    Parameters:
    FileName (str): The name of the file to load.

    Returns:
    object: The loaded data from the file.
    """
    
    # Load the results
    with open(FileName, 'rb') as f:
        localFile = pickle.load(f)

    return localFile


def createModelTxTFiles(rounds, threshold, threshold2, allUndirectedEdges, roundsAccuracies, roundsAccuracies2):
    """
    Creates and saves various text files based on the given parameters.

    Parameters:
    - rounds (int): The number of rounds.
    - threshold (float): The threshold value.
    - threshold2 (float): The second threshold value.
    - allUndirectedEdges (dict): A dictionary containing undirected edges.
    - roundsAccuracies (list): A list of accuracies for each round.
    - roundsAccuracies2 (list): A list of accuracies for each round (using the second threshold).

    Returns:
    None
    """

    # Save allEdges to a txt file
    with open('allEdges.txt', 'w') as f:
        f.write('(node_id1, node_id2): numOfRemoved numOfRecoveries_T9 numOfRecoveries_T95\n')
        for key, edge in allUndirectedEdges.items():
            f.write(f'{key}: {edge.numOfRemoved} {edge.numOfRecoveries_T9} {edge.numOfRecoveries_T95}\n')

    # Save results to a txt file
    resultsFileName = 'results_R={}_T={}.txt'.format(rounds, threshold)

    with open(resultsFileName, 'w') as f:
        for key, edge in allUndirectedEdges.items():
            if edge.numOfRemoved > 0:
                f.write(f'{edge.numOfRecoveries_T95}\n')

    # Save results to a txt file
    resultsFileName = 'results_R={}_T={}.txt'.format(rounds, threshold2)

    with open(resultsFileName, 'w') as f:
        for key, edge in allUndirectedEdges.items():
            if edge.numOfRemoved > 0:
                f.write(f'{edge.numOfRecoveries_T9}\n')

    # Save roundsAccuracies to a txt file
    roundsAccuraciesFileName = 'roundsAccuracies_R={}_T={}.txt'.format(rounds, threshold)

    with open(roundsAccuraciesFileName, 'w') as f:
        for item in roundsAccuracies:
            f.write("%s\n" % item)

    # Save roundsAccuracies to a txt file
    roundsAccuraciesFileName = 'roundsAccuracies_R={}_T={}.txt'.format(rounds, threshold2)

    with open(roundsAccuraciesFileName, 'w') as f:
        for item in roundsAccuracies2:
            f.write("%s\n" % item)


