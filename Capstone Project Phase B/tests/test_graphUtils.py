import unittest
import numpy as np
from Edge import Edge
from graphUtils import buildDictOfAllEdges , prepare_data
import torch
from unittest.mock import Mock, patch
import unittest


class TestDictOfAllEdges(unittest.TestCase):
    def setUp(self):

        self.edge_index_undirected = np.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        self.result_dict_undirected = buildDictOfAllEdges(self.edge_index_undirected)        
        # expected result_dict_undirected:
        # (0,1) : edgeDirectOne = 0, edgeDirectTwo = 2
        # (0,2) : edgeDirectOne = 1, edgeDirectTwo = 4
        # (1,2) : edgeDirectOne = 3, edgeDirectTwo = 5


    def test_none(self):
        """
        Test case for the 'buildDictOfAllEdges' method.

        This test case verifies the behavior of the 'buildDictOfAllEdges' function when None is provided.
        It checks if an AttributeError is raised.

        """
        edge_index = None
        with self.assertRaises(AttributeError):
            result = buildDictOfAllEdges(edge_index)

    def test_empty(self):
        """
        Test case to verify the behavior of buildDictOfAllEdges when the edge_index is empty.
        """
        edge_index = np.array([[], []])
        result = buildDictOfAllEdges(edge_index)
        self.assertEqual(result, {})
    
    def test_single_edge(self):
        """
        Test case for the 'test_single_edge' method.

        This test case verifies the behavior of the 'buildDictOfAllEdges' function when a single edge is provided.
        It checks if the resulting dictionary has a length of 1, and if the edge object in the dictionary has the expected attributes.

        """
        edge_index = np.array([[0], [1]])
        result = buildDictOfAllEdges(edge_index)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[(0, 1)], Edge)
        self.assertEqual(result[(0, 1)].edgeDirectOne, 0)
        self.assertEqual(result[(0, 1)].edgeDirectTwo, 0)

    def test_multiple_edges_directed(self):
        """
        Test case for the `test_multiple_edges_directed` method.

        This test case checks if a `KeyError` is raised when calling the `buildDictOfAllEdges` function
        with a multiple edges directed graph represented by the `edge_index` array.

        """
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])

        with self.assertRaises(KeyError):
            result = buildDictOfAllEdges(edge_index)


    def test_buildDictOfAllEdges_size_undirected(self):
        """
        Test case to check the size of the result_dict after calling the buildDictOfAllEdges method.
        """
        self.assertEqual(len(self.result_dict_undirected), 3)

    def test_buildDictOfAllEdges_values_undirected(self):
        """
        Test case to check the values of the result_dict after calling the buildDictOfAllEdges method.
        """
        self.assertEqual(self.result_dict_undirected[(0,1)].edgeDirectOne,0)
        self.assertEqual(self.result_dict_undirected[(0,1)].edgeDirectTwo,2)
        self.assertEqual(self.result_dict_undirected[(0,2)].edgeDirectOne,1)
        self.assertEqual(self.result_dict_undirected[(0,2)].edgeDirectTwo,4)
        self.assertEqual(self.result_dict_undirected[(1,2)].edgeDirectOne,3)
        self.assertEqual(self.result_dict_undirected[(1,2)].edgeDirectTwo,5)

class TestPrepareData(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment by initializing necessary objects and mocks.

        This method is executed before each test case.

        Args:
            self: The test case object.

        Returns:
            None
        """
        self.dm = Mock()
        self.dm.dataset.data.num_nodes = 4
        self.dm.dataset.data.edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [1, 3, 0, 2, 1, 3, 0, 2]])

        ### Mocking the Edge class edge (0, 1)##############
        self.edge_mock = Mock()
        self.edge_mock.edgeDirectOne = 0
        self.edge_mock.edgeDirectTwo = 2
        self.edge_mock.isRemoveInThisRound = False
        self.edge_mock.numOfRecoveries_T95 = 0
        self.edge_mock.numOfRecoveries_T9 = 0
        self.edge_mock.numOfRemoved = 0
        ######################################################

        ### Mocking the Edge class edge (0,3)##############
        self.edge_mock2 = Mock()
        self.edge_mock2.edgeDirectOne = 1
        self.edge_mock2.edgeDirectTwo = 6
        self.edge_mock2.isRemoveInThisRound = False
        self.edge_mock2.numOfRecoveries_T95 = 0
        self.edge_mock2.numOfRecoveries_T9 = 0
        self.edge_mock2.numOfRemoved = 0
        ######################################################

        ### Mocking the Edge class edge (1,2)##############
        self.edge_mock3 = Mock()
        self.edge_mock3.edgeDirectOne = 3
        self.edge_mock3.edgeDirectTwo = 4
        self.edge_mock3.isRemoveInThisRound = False
        self.edge_mock3.numOfRecoveries_T95 = 0
        self.edge_mock3.numOfRecoveries_T9 = 0
        self.edge_mock3.numOfRemoved = 0
        ######################################################

        ### Mocking the Edge class edge (2,3)##############
        self.edge_mock4 = Mock()
        self.edge_mock4.edgeDirectOne = 5
        self.edge_mock4.edgeDirectTwo = 7
        self.edge_mock4.isRemoveInThisRound = False
        self.edge_mock4.numOfRecoveries_T95 = 0
        self.edge_mock4.numOfRecoveries_T9 = 0
        self.edge_mock4.numOfRemoved = 0
        ######################################################

        ### Mocking allUndirectedEdges attribute ##############
        self.dm.allUndirectedEdges = {(0, 1): self.edge_mock, (0, 3): self.edge_mock2, (1, 2): self.edge_mock3, (2, 3): self.edge_mock4}
        ####################################################################################

    def test_prepare_data_with_None_values(self):
        """
        Test case to verify the behavior of the prepare_data function when it receives None values as inputs.
        """

        # Call the function with None inputs
        with self.assertRaises(AttributeError):
            prepare_data(None)

    def test_prepare_data_no_nodes(self):
        self.dm.dataset.data.num_nodes = 0

        with self.assertRaises(ValueError):  
            prepare_data(self.dm)

    def test_prepare_data_train_test_split(self):
        """
        Test case to verify that the train/test split is done correctly.
        """

        prepare_data(self.dm)
        total_nodes = self.dm.dataset.data.num_nodes
        num_train_nodes = int(total_nodes * 0.7)
        num_test_nodes = total_nodes - num_train_nodes

        self.assertEqual(len(self.dm.train_node_indices), num_train_nodes)
        self.assertEqual(len(self.dm.test_node_indices), num_test_nodes)

    def test_prepare_data_shuffle(self):
        """
        Test case to verify that random.shuffle is called with the correct arguments.
        """

        with patch('random.shuffle') as mock_shuffle:
            prepare_data(self.dm)
            total_nodes = self.dm.dataset.data.num_nodes
            expected_argument = list(range(total_nodes))
            mock_shuffle.assert_called_once_with(expected_argument)

    def test_prepare_data_edge_assignment(self):
        """
        Test case to verify that edges are correctly assigned to the training and testing sets.
        """

        prepare_data(self.dm)

        # Check that all edges in the training set have both nodes in the training set
        for edge in self.dm.dictTrainEdges.values():
            self.assertIn(edge[0], self.dm.train_node_indices)
            self.assertIn(edge[1], self.dm.train_node_indices)

        # Check that all edges in the testing set have both nodes in the testing set
        for edge in self.dm.dictTestEdges.values():
            self.assertIn(edge[0], self.dm.test_node_indices)
            self.assertIn(edge[1], self.dm.test_node_indices)

if __name__ == '__main__':
    unittest.main()