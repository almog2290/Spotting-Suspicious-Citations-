import unittest
from fileUtils import createModelTxTFiles
from unittest.mock import Mock
import unittest
import os
import re

class TestCreateModelTxTFiles(unittest.TestCase):
    def setUp(self):
        self.rounds = 3
        self.threshold = 0.95
        self.threshold2 = 0.9
        
        ### Mocking the Edge class edge (3, 2544)##############
        self.edge_mock = Mock()
        self.edge_mock.edgeDirectOne = 11
        self.edge_mock.edgeDirectTwo = 10222
        self.edge_mock.isRemoveInThisRound = True
        self.edge_mock.numOfRecoveries_T95 = 1
        self.edge_mock.numOfRecoveries_T9 = 2
        self.edge_mock.numOfRemoved = 3
        ######################################################

        ### Mocking the Edge class edge (6, 1602)##############
        self.edge_mock2 = Mock()
        self.edge_mock2.edgeDirectOne = 23
        self.edge_mock2.edgeDirectTwo = 6315
        self.edge_mock2.isRemoveInThisRound = True
        self.edge_mock2.numOfRecoveries_T95 = 2
        self.edge_mock2.numOfRecoveries_T9 = 2
        self.edge_mock2.numOfRemoved = 3
        ######################################################

        ### Mocking allUndirectedEdges attribute ##############
        self.allUndirectedEdges = {(3, 2544): self.edge_mock, (6, 1602): self.edge_mock2}
        ####################################################################################

        self.roundsAccuracies = [0.8, 0.9, 0.7]  
        self.roundsAccuracies2 = [0.85, 0.92, 0.82]

        createModelTxTFiles(self.rounds, self.threshold, self.threshold2, self.allUndirectedEdges, self.roundsAccuracies, self.roundsAccuracies2)

    def test_files_are_created(self):
        """
        Test case to check if the required files are created.

        This method calls the createModelTxTFiles function with the necessary parameters
        and then checks if the expected files are created.

        Returns:
            None
        """

        self.assertTrue(os.path.exists('allEdges.txt'))
        self.assertTrue(os.path.exists('results_R={}_T={}.txt'.format(self.rounds, self.threshold)))
        self.assertTrue(os.path.exists('results_R={}_T={}.txt'.format(self.rounds, self.threshold2)))
        self.assertTrue(os.path.exists('roundsAccuracies_R={}_T={}.txt'.format(self.rounds, self.threshold)))
        self.assertTrue(os.path.exists('roundsAccuracies_R={}_T={}.txt'.format(self.rounds, self.threshold2)))


    def test_file_content_allEdges(self):
        """
        Test the content of the 'allEdges.txt' file.

        This method reads the lines from the 'allEdges.txt' file and checks if each line matches a specific pattern.
        The pattern is defined as '(node_id1, node_id2): numOfRemoved numOfRecoveries_T9 numOfRecoveries_T95\n'.

        If a line does not match the expected format, an assertion error is raised.

        Note: The header or comment line is skipped during the matching process.

        Raises:
            AssertionError: If a line does not match the expected format.

        """
        with open('allEdges.txt', 'r') as file:
            lines = file.readlines()

        # Define the pattern that each line should match
        pattern = re.compile(r'\(\d+, \d+\): \d+ \d+ \d+\n')

        for line in lines:
            # Skip the header or comment line
            if line.startswith('(node_id1, node_id2):'):
                continue

            self.assertTrue(pattern.match(line), f"Line does not match expected format: {line}")


    def test_file_content_results(self):
        """
        Test the content of the file 'results_R_T=0.9' to ensure it matches the expected format.

        This test reads the lines from the file and checks if each line matches a specific pattern.
        It also verifies that the number of lines in the file matches the number of edges in the 'allUndirectedEdges' list.

        Raises:
            AssertionError: If a line does not match the expected format or the number of lines does not match the number of edges.
        """
        with open('results_R={}_T={}.txt'.format(self.rounds, self.threshold2), 'r') as file:
            lines = file.readlines()

        # Define the pattern that each line should match
        pattern = re.compile(r'\d+\n')

        for line in lines:
            self.assertTrue(pattern.match(line), f"Line does not match expected format: {line}")

        # Check the number of lines
        self.assertEqual(len(lines), len(self.allUndirectedEdges), "Number of lines does not match number of edges")


    def test_file_content_results_095(self):
        """
        Test the content of the 'results_R_T=0.95' file.

        This method reads the lines from the 'results_R_T=0.95' file and checks if each line matches the expected format.
        It also checks if the number of lines in the file matches the number of edges in the 'allUndirectedEdges' list.

        Raises:
            AssertionError: If a line does not match the expected format or the number of lines does not match the number of edges.
        """
        with open('results_R={}_T={}.txt'.format(self.rounds, self.threshold), 'r') as file:
            lines = file.readlines()

        # Define the pattern that each line should match
        pattern = re.compile(r'\d+\n')

        for line in lines:
            self.assertTrue(pattern.match(line), f"Line does not match expected format: {line}")

        # Check the number of lines
        self.assertEqual(len(lines), len(self.allUndirectedEdges), "Number of lines does not match number of edges")


    def test_file_content_roundsAccuracies(self):
        """
        Test the content of the file 'roundsAccuracies_R_T=0.9'.

        This method reads the lines from the file and checks if each line matches the expected format.
        It also checks if the number of lines in the file matches the number of rounds.

        Raises:
            AssertionError: If a line does not match the expected format or the number of lines does not match the number of rounds.
        """
        with open('roundsAccuracies_R={}_T={}.txt'.format(self.rounds, self.threshold2), 'r') as file:
            lines = file.readlines()

        # Define the pattern that each line should match
        pattern = re.compile(r'\d+\.\d+\n')

        for line in lines:
            self.assertTrue(pattern.match(line), f"Line does not match expected format: {line}")

        # Check the number of lines
        self.assertEqual(len(lines), self.rounds, "Number of lines does not match number of rounds")


    def test_file_content_roundsAccuracies_095(self):
        """
        Test method to check the content of the 'roundsAccuracies_R_T=0.95' file.

        This method reads the lines from the file and checks if each line matches the expected format.
        It also checks if the number of lines in the file matches the number of rounds.

        Raises:
            AssertionError: If a line does not match the expected format or the number of lines does not match the number of rounds.
        """
        with open('roundsAccuracies_R={}_T={}.txt'.format(self.rounds, self.threshold), 'r') as file:
            lines = file.readlines()

        # Define the pattern that each line should match
        pattern = re.compile(r'\d+\.\d+\n')

        for line in lines:
            self.assertTrue(pattern.match(line), f"Line does not match expected format: {line}")

        # Check the number of lines
        self.assertEqual(len(lines), self.rounds, "Number of lines does not match number of rounds")

if __name__ == '__main__':
    unittest.main()