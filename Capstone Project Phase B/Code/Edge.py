
class Edge:
    """
    Represents an edge in a graph.

    Attributes:
        edgeDirectOne (int): The index of the edge in the first direction in edge_index.
        edgeDirectTwo (int): The index of the edge in the second direction in edge_index.
        numOfRecoveries_T95 (int): The number of recoveries at 95% confidence level.
        numOfRecoveries_T9 (int): The number of recoveries at 90% confidence level.
        numOfRemoved (int): The number of times the specific edge has been removed.
        isRemoveInThisRound (bool): Indicates if the edge is removed in the current round.
    """

    def __init__(self, edgeDirectOne, edgeDirectTwo):
        self.edgeDirectOne = edgeDirectOne
        self.edgeDirectTwo = edgeDirectTwo
        self.numOfRecoveries_T95 = 0
        self.numOfRecoveries_T9 = 0
        self.numOfRemoved = 0
        self.isRemoveInThisRound = False

