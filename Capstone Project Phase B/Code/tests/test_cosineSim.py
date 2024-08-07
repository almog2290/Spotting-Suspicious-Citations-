import unittest
from cosineSim import padVectors , calc_cosine_similarity , createGmaeVectors , calc_edges_restore
import torch
from unittest.mock import Mock, patch , MagicMock
import unittest


class TestPadVectors(unittest.TestCase):
    def test_same_length(self):
        """
        Test case to verify the behavior of the padVectors function when given two vectors of the same length.
        """
        vector1 = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        vector2 = torch.tensor([[4, 5, 6]], dtype=torch.float32)
        result1, result2 = padVectors(vector1, vector2)
        torch.testing.assert_close(result1, vector1)
        torch.testing.assert_close(result2, vector2)

    def test_vector1_longer(self):
        """
        Test case to verify the behavior of the padVectors function when vector1 is longer than vector2.
        """
        vector1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
        vector2 = torch.tensor([[5, 6, 7]], dtype=torch.float32)
        result1, result2 = padVectors(vector1, vector2)
        torch.testing.assert_close(result1, vector1)
        torch.testing.assert_close(result2, torch.tensor([[5, 6, 7, 0]], dtype=torch.float32))

    def test_vector2_longer(self):
        """
        Test case to verify the behavior of the padVectors function when the second vector is longer than the first vector.
        """
        vector1 = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        vector2 = torch.tensor([[4, 5, 6, 7]], dtype=torch.float32)
        result1, result2 = padVectors(vector1, vector2)
        torch.testing.assert_close(result1, torch.tensor([[1, 2, 3, 0]], dtype=torch.float32))
        torch.testing.assert_close(result2, vector2)

    def test_empty_vectors(self):
        """
        Test case for handling empty input vectors.

        This test verifies that the `padVectors` function correctly handles the scenario
        where both input vectors are empty. It checks if the function returns the original
        empty vectors without any modifications.

        Returns:
            None
        """
        vector1 = torch.tensor([[]], dtype=torch.float32)
        vector2 = torch.tensor([[]], dtype=torch.float32)
        result1, result2 = padVectors(vector1, vector2)
        torch.testing.assert_close(result1, vector1)
        torch.testing.assert_close(result2, vector2)

class TestCalcCosineSimilarity(unittest.TestCase):
    def test_orthogonal_vectors(self):
        """
        Test case to check the cosine similarity between two orthogonal vectors.

        This test case creates two orthogonal vectors, calculates the cosine similarity
        between them using the `calc_cosine_similarity` function, and asserts that the
        result is equal to 0.

        """
        vector1 = torch.tensor([[1, 0]], dtype=torch.float32)
        vector2 = torch.tensor([[0, 1]], dtype=torch.float32)
        result = calc_cosine_similarity(vector1, vector2)
        self.assertEqual(result, 0)

    def test_identical_vectors(self):
        """
        Test case to check the cosine similarity calculation for identical vectors.

        This test case creates two identical vectors and calculates the cosine similarity between them.
        The expected result is 1, as identical vectors have a cosine similarity of 1.

        """
        vector1 = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        vector2 = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        result = calc_cosine_similarity(vector1, vector2)
        self.assertEqual(round(result), 1)

    def test_non_orthogonal_non_identical_vectors(self):
        """
        Test case to check the cosine similarity calculation for non-orthogonal and non-identical vectors.
        """
        vector1 = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        vector2 = torch.tensor([[4, 5, 6]], dtype=torch.float32)
        result = calc_cosine_similarity(vector1, vector2)
        expected = torch.nn.functional.cosine_similarity(vector1, vector2).item()
        self.assertEqual(result, expected)

class TestCreateGmaeVectors(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment by mocking the model, datamodule, and batched_data.
        EgoGraph Shape [1 graph , nodes , 64 features] will be replace in the mock to size of [1, 4, 4] for simplicity. 
        """
        # Mock model and datamodule
        self.model = Mock()
        self.model.device = 'cpu'  # Mock device
        self.dm = Mock()

        # Mock batched_data
        self.batched_data1 = MagicMock()
        self.batched_data1.x = torch.tensor([[[0.7677, 0.8620, 0.0227, 0.8534],
                                            [0.8205, 0.4154, 0.6059, 0.4076],
                                            [0.7679, 0.3890, 0.7005, 0.0920],
                                            [0.8953, 0.9698, 0.1034, 0.3479]]])
        self.batched_data1.node_id.__getitem__().item.return_value = 1
        self.batched_data1.to.return_value = self.batched_data1

        self.batched_data2 = MagicMock()
        self.batched_data2.x = torch.tensor([[[0.6746, 0.1288, 0.4886, 0.3104],
                                            [0.0117, 0.0418, 0.7154, 0.9269],
                                            [0.9016, 0.4975, 0.7908, 0.5868],
                                            [0.6773, 0.1513, 0.9743, 0.2930]]])
        self.batched_data2.node_id.__getitem__().item.return_value = 2
        self.batched_data2.to.return_value = self.batched_data2

        # Mock predict_dataloader() method
        self.dm.predict_dataloader.return_value = [self.batched_data1, self.batched_data2]

        # Mock generate_pretrain_embeddings_for_downstream_task2() method
        self.model.generate_pretrain_embeddings_for_downstream_task2.side_effect = lambda bd: bd.x

        # Test createGmaeVectors function
        self.gmaeVectors = createGmaeVectors(self.model, self.dm)


    def test_createGmaeVectors_with_null_model(self):
        """
        Test case to verify the behavior of createGmaeVectors function when the model is None.

        This test case checks if an AttributeError is raised when the createGmaeVectors function is called with a None model.
        The expected behavior is that an AttributeError should be raised.

        """
        self.model = None

        try:
            result = createGmaeVectors(self.model, self.dm)
            assert False, "Expected an exception but it wasn't raised"
        except AttributeError:
            pass  # Expected this exception

    def test_createGmaeVectors_with_null_dm(self):
        """
        Test case to verify the behavior of createGmaeVectors function when dm is None.

        This test case checks if an AttributeError is raised when the dm parameter is set to None.
        The function should raise an exception in this case.

        """
        self.dm = None

        try:
            result = createGmaeVectors(self.model, self.dm)
            assert False, "Expected an exception but it wasn't raised"
        except AttributeError:
            pass  # Expected this exception

    def test_keys_in_GAMEVectors_dictionary(self):
        """
        Test case to check if all keys are present in the GAMEVectors(egoGraph's) dictionary.

        It asserts that the length of the dictionary is 2 and checks if keys 1 and 2 are present.
        """
        self.assertEqual(len(self.gmaeVectors), 2)
        self.assertTrue(1 in self.gmaeVectors)
        self.assertTrue(2 in self.gmaeVectors)

    def test_values_in_GAMEVectors_dictionary(self):
        """
        Test case to check if the values in the GAMEVectors(egoGraph's) dictionary are the mean of the 'x'[0] tensor for each egoGraph.
        """
        # Assert that the values in the dictionary are the mean of the 'x'[0] tensor for each batched_data
        self.assertTrue(torch.allclose(self.gmaeVectors[1], self.batched_data1.x[0].mean(dim=0, keepdim=True)))
        self.assertTrue(torch.allclose(self.gmaeVectors[2], self.batched_data2.x[0].mean(dim=0, keepdim=True)))

    def test_values_shape_in_GAMEVectors(self):
        """
        Test case to check the shape of values in the GAMEVectors(egoGraph's) dictionary.

        This method asserts that the values in the dictionary have the correct shape.
        It compares the shape of each value with the expected shape using the `assertEqual` method.

        Returns:
            None
        """
        # Assert that the values in the dictionary have the correct shape
        self.assertEqual(self.gmaeVectors[1].shape, (1, 4))
        self.assertEqual(self.gmaeVectors[2].shape, (1, 4))

class TestCalcEdgesRestore(unittest.TestCase):
    def setUp(self):

        self.edge_dict = {10: (3, 2544) , 22: (6 , 1602) }   # edge_dict = {edge_index: (node1, node2)}
        self.gmaeVectors = {3: None , 6: None , 1602: None , 2544: None}  # gmaeVectors = {node: tensor [1,4] (egograph representation)}
        self.gmaeVectors[3] = torch.tensor([[0.2338, 0.6381, 0.9893, 0.9771]], dtype=torch.float32)
        self.gmaeVectors[2544] = torch.tensor([[0.2338, 0.6381, 0.9893, 0.9771]], dtype=torch.float32)
        self.gmaeVectors[6] = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
        self.gmaeVectors[1602] = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)

        ### Mocking the Edge class edge (3, 2544)##############
        self.edge_mock = Mock()
        self.edge_mock.edgeDirectOne = 11
        self.edge_mock.edgeDirectTwo = 10222
        self.edge_mock.isRemoveInThisRound = True
        self.edge_mock.numOfRecoveries_T95 = 0
        self.edge_mock.numOfRecoveries_T9 = 0
        self.edge_mock.numOfRemoved = 1
        ######################################################

        ### Mocking the Edge class edge (6, 1602)##############
        self.edge_mock2 = Mock()
        self.edge_mock2.edgeDirectOne = 23
        self.edge_mock2.edgeDirectTwo = 6315
        self.edge_mock2.isRemoveInThisRound = True
        self.edge_mock2.numOfRecoveries_T95 = 0
        self.edge_mock2.numOfRecoveries_T9 = 0
        self.edge_mock2.numOfRemoved = 1
        ######################################################

        ### Mocking the DataModule class and its allUndirectedEdges attribute ##############
        self.dm = Mock()
        self.dm.allUndirectedEdges = {(3, 2544): self.edge_mock, (6, 1602): self.edge_mock2}
        ####################################################################################

        # Mocking the calc_cosine_similarity function
        self.patcher = patch('cosineSim.calc_cosine_similarity')
        self.mock_cosine_similarity = self.patcher.start()
        def side_effect(*args):
            if torch.all(args[0] == self.gmaeVectors[3]) and torch.all(args[1] == self.gmaeVectors[2544]):
                return 1
            elif torch.all(args[0] == self.gmaeVectors[6]) and torch.all(args[1] == self.gmaeVectors[1602]):
                return 1
            
        self.mock_cosine_similarity.side_effect = side_effect

    #TODO: accuracy calculation division by zero , fix in calc_edges_restore
    def test_isRemoveInThisRound_False(self):
        """
        Test case to verify the behavior of calc_edges_restore when isRemoveInThisRound is False.

        This test sets the isRemoveInThisRound attribute of two edge_mock objects to False,
        and then calls the calc_edges_restore function with the given edge_dict, gmaeVectors,
        threshold, and dm. The expected result is 0, indicating that no recovery happens when
        isRemoveInThisRound is False.
        """
        self.edge_mock.isRemoveInThisRound = False
        self.edge_mock2.isRemoveInThisRound = False
        threshold = 0.95
        accResult = calc_edges_restore(self.edge_dict, self.gmaeVectors, threshold, self.dm)
        self.assertEqual(accResult, 0)  # Assuming no recovery happens if isRemoveInThisRound is False

    def test_isRemoveInThisRound_True_edgeThreshold_lessThan_0_9(self):
        """
        Test case to check if isRemoveInThisRound is True when edgeThreshold is less than 0.9.

        This test case sets up the necessary conditions and checks if the `calc_edges_restore` function returns the expected result , 
        when the `isRemoveInThisRound` attribute is True for two edges and the `threshold` is set to 0.9.

        The test case defines a new side effect for the `cosine_similarity` function mock to return different cosine similarity values for specific input vectors. 
        It then calls the `calc_edges_restore` function with the necessary arguments and asserts that the result is equal to 0, 
        assuming that no recovery happens if the threshold is less than 0.9.
        """
        self.edge_mock.isRemoveInThisRound = True
        self.edge_mock2.isRemoveInThisRound = True
        threshold = 0.9

        # Define a new side effect for cosine similarity edge (3, 2544) and (6, 1602)
        def new_side_effect(*args):
            if torch.all(args[0] == self.gmaeVectors[3]) and torch.all(args[1] == self.gmaeVectors[2544]):
                return 0.8
            elif torch.all(args[0] == self.gmaeVectors[6]) and torch.all(args[1] == self.gmaeVectors[1602]):
                return 0.85

        # Assign the new side effect function to the mock
        self.mock_cosine_similarity.side_effect = new_side_effect

        accResult = calc_edges_restore(self.edge_dict, self.gmaeVectors, threshold, self.dm)
        self.assertEqual(accResult, 0)  # Assuming no recovery happens if threshold < 0.9


    def test_isRemoveInThisRound_True_edgeThreshold_lessThan_0_95(self):
        """
        Test case to check if isRemoveInThisRound is True and edgeThreshold is less than 0.95.

        This test case defines a new side effect for cosine similarity edge (3, 2544) and (6, 1602),
        and assigns the new side effect function to the mock. It then calls the `calc_edges_restore`
        function with the given edge dictionary, GMAE vectors, threshold, and dm. Finally, it asserts
        that the result is equal to 0, assuming no recovery happens if the threshold is less than 0.95.
        """
        self.edge_mock.isRemoveInThisRound = True
        self.edge_mock2.isRemoveInThisRound = True
        threshold = 0.95

        # Define a new side effect for cosine similarity edge (3, 2544) and (6, 1602)
        def new_side_effect(*args):
            if torch.all(args[0] == self.gmaeVectors[3]) and torch.all(args[1] == self.gmaeVectors[2544]):
                return 0.8
            elif torch.all(args[0] == self.gmaeVectors[6]) and torch.all(args[1] == self.gmaeVectors[1602]):
                return 0.85

        # Assign the new side effect function to the mock
        self.mock_cosine_similarity.side_effect = new_side_effect

        accResult = calc_edges_restore(self.edge_dict, self.gmaeVectors, threshold, self.dm)
        self.assertEqual(accResult, 0)  # Assuming no recovery happens if threshold < 0.95

    def test_isRemoveInThisRound_True_edgeThreshold_greaterThan_0_95(self):
        """
        Test case to verify the behavior of the `calc_edges_restore` function when `isRemoveInThisRound` is True,
        and the `edgeThreshold` is greater than 0.95.

        The test sets up a mock object for cosine similarity and defines a new side effect for specific input arguments.
        It then calls the `calc_edges_restore` function with the provided test data and asserts that the result is equal to 1,
        assuming that all edges are recovered if the threshold is greater than 0.95.
        """

        self.edge_mock.isRemoveInThisRound = True
        self.edge_mock2.isRemoveInThisRound = True
        threshold = 0.95

        # Define a new side effect for cosine similarity edge (3, 2544) and (6, 1602)
        def new_side_effect(*args):
            if torch.all(args[0] == self.gmaeVectors[3]) and torch.all(args[1] == self.gmaeVectors[2544]):
                return 0.96
            elif torch.all(args[0] == self.gmaeVectors[6]) and torch.all(args[1] == self.gmaeVectors[1602]):
                return 0.98

        # Assign the new side effect function to the mock
        self.mock_cosine_similarity.side_effect = new_side_effect

        accResult = calc_edges_restore(self.edge_dict, self.gmaeVectors, threshold, self.dm)
        self.assertEqual(accResult, 1)  # Assuming all edges are recovered if threshold > 0.95

    def test_isRemoveInThisRound_True_edgeThreshold_greaterThan_0_9(self):
        """
        Test case to check if isRemoveInThisRound is True and edgeThreshold is greater than 0.9.

        This test case sets up the necessary conditions for the calculation of edges restoration.
        It defines a new side effect for cosine similarity edge (3, 2544) and (6, 1602) and assigns
        it to the mock_cosine_similarity function. The test then calls the calc_edges_restore
        function with the given edge dictionary, gmaeVectors, threshold, and dm. Finally, it asserts
        that the result is equal to 1, assuming that all edges are recovered if the threshold is
        greater than 0.9.
        """
    
        self.edge_mock.isRemoveInThisRound = True
        self.edge_mock2.isRemoveInThisRound = True
        threshold = 0.9

        # Define a new side effect for cosine similarity edge (3, 2544) and (6, 1602)
        def new_side_effect(*args):
            if torch.all(args[0] == self.gmaeVectors[3]) and torch.all(args[1] == self.gmaeVectors[2544]):
                return 0.92
            elif torch.all(args[0] == self.gmaeVectors[6]) and torch.all(args[1] == self.gmaeVectors[1602]):
                return 0.93

        # Assign the new side effect function to the mock
        self.mock_cosine_similarity.side_effect = new_side_effect

        accResult = calc_edges_restore(self.edge_dict, self.gmaeVectors, threshold, self.dm)
        self.assertEqual(accResult, 1)  # Assuming all edges are recovered if threshold > 0.9

    def test_calc_of_accuracy_multiEdges(self):
        """
        Test case for calculating accuracy with multiple edges.

        This test case verifies the accuracy calculation when there are multiple edges to be considered.
        It defines a new side effect for the cosine similarity edge (3, 2544) and (6, 1602) and assigns it to the mock.
        Then, it calls the `calc_edges_restore` function with the given edge dictionary, GMAE vectors, threshold, and dm.
        Finally, it asserts that the calculated accuracy result is equal to 0.5.
        """
        threshold = 0.95

        # Define a new side effect for cosine similarity edge (3, 2544) and (6, 1602)
        def new_side_effect(*args):
            if torch.all(args[0] == self.gmaeVectors[3]) and torch.all(args[1] == self.gmaeVectors[2544]):
                return 0.96
            elif torch.all(args[0] == self.gmaeVectors[6]) and torch.all(args[1] == self.gmaeVectors[1602]):
                return 0.85

        # Assign the new side effect function to the mock
        self.mock_cosine_similarity.side_effect = new_side_effect

        accResult = calc_edges_restore(self.edge_dict, self.gmaeVectors, threshold, self.dm)
        self.assertEqual(accResult, 0.5)

    def test_calc_edges_restore_with_None_values(self):
        """
        Test case to verify the behavior of calc_edges_restore function when it receives None values as inputs.
        """

        # Define a new side effect for cosine similarity that returns None
        def new_side_effect(*args):
            return None

        # Assign the new side effect function to the mock
        self.mock_cosine_similarity.side_effect = new_side_effect

        # Call the function with normal inputs
        with self.assertRaises(TypeError):
            calc_edges_restore(self.edge_dict, self.gmaeVectors, 0.95, self.dm)

        # Call the function with None inputs
        with self.assertRaises(ValueError):
            calc_edges_restore(None, self.gmaeVectors, 0.95, self.dm)
        with self.assertRaises(ValueError):
            calc_edges_restore(self.edge_dict, None, 0.95, self.dm)
        with self.assertRaises(ValueError):
            calc_edges_restore(self.edge_dict, self.gmaeVectors, 0.95, None)

if __name__ == '__main__':
    unittest.main()
