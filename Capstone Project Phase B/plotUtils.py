import matplotlib.pyplot as plt
import numpy as np
from fileUtils import saveToLocalFile  

def plot_correct_edges(rounds, correct_edges_final, threshold, roundsAccuracies):
    """
    Plot the distribution of restored edges and save the results.

    Parameters:
    - rounds (int): The number of rounds.
    - correct_edges_final (numpy.ndarray): An array containing the number of correct edges for each round.
    - threshold (float): The threshold value.
    - roundsAccuracies (list): A list of accuracies for each round.

    Returns:
    None
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Count correct edges
    cnt_correct_edges = np.zeros(rounds+1 , dtype=int)
    for i in range(rounds+1):
        cnt_correct_edges[i] = np.count_nonzero(correct_edges_final == i)

    print('######### final results #######################')
    print('cnt_correct_edges ->',cnt_correct_edges)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(cnt_correct_edges)), cnt_correct_edges ,color='green' , width=0.7)
    plt.xlabel('Amount of restored edges')
    plt.ylabel('Number of edges')
    plt.title('Distribution of restored edges [ rounds ={} , threshold ={}]'.format(rounds,threshold))
    plt.savefig('histogramCorrect_R={}_T={}.png'.format(rounds,threshold))

    # Count average accuracy
    avgAcc = sum(roundsAccuracies)/len(roundsAccuracies)
    print('avgAcc ->',avgAcc)

    # Save cnt_correct_edges
    fileName='cnt_correct_edges_R={}_T={}.pkl'.format(rounds,threshold)
    saveToLocalFile(fileName,cnt_correct_edges)