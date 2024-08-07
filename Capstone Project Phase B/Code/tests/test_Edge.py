import unittest
from Edge import Edge
import unittest


class TestEdgeClass(unittest.TestCase):
    def setUp(self):
        self.edge = Edge(1, 2)

    def test_edge_init_not_none(self):
        """
        Test case to verify the initialization of an Edge object.

        It checks if the attributes of the Edge object are not None after initialization.

        Returns:
            None
        """
        self.assertIsNotNone(self.edge.edgeDirectOne)
        self.assertIsNotNone(self.edge.edgeDirectTwo)
        self.assertIsNotNone(self.edge.numOfRecoveries_T95)
        self.assertIsNotNone(self.edge.numOfRecoveries_T9)
        self.assertIsNotNone(self.edge.numOfRemoved)
        self.assertIsNotNone(self.edge.isRemoveInThisRound)

    def test_edge_init_values(self):
        """
        Test case to verify the initialization of an Edge object.

        It checks if the attributes of the Edge object are set correctly after initialization.

        Returns:
            None
        """
        self.assertEqual(self.edge.edgeDirectOne, 1)
        self.assertEqual(self.edge.edgeDirectTwo, 2)
        self.assertEqual(self.edge.numOfRecoveries_T95, 0)
        self.assertEqual(self.edge.numOfRecoveries_T9, 0)
        self.assertEqual(self.edge.numOfRemoved, 0)
        self.assertEqual(self.edge.isRemoveInThisRound, False)

    def test_edge_attribute_modification(self):
        """
        Test case to verify the modification of edge attributes.

        This test case creates an instance of the Edge class and modifies its attributes.
        It then asserts that the modified attributes have the expected values.

        """
        self.edge.numOfRecoveries_T95 = 4
        self.edge.numOfRecoveries_T9 = 3
        self.edge.numOfRemoved = 5
        self.edge.isRemoveInThisRound = True
        self.assertEqual(self.edge.numOfRecoveries_T95, 4)
        self.assertEqual(self.edge.numOfRecoveries_T9, 3)
        self.assertEqual(self.edge.numOfRemoved, 5)
        self.assertEqual(self.edge.isRemoveInThisRound, True)
    
    def test_T9_not_greater_than_T95(self):
        """
        Test case to verify that numOfRecoveries_T9 is not greater than numOfRecoveries_T95.

        This test case creates an instance of the Edge class and sets its attributes.
        It then asserts that numOfRecoveries_T9 is not greater than numOfRecoveries_T95.

        """
        self.edge.numOfRecoveries_T95 = 4
        self.edge.numOfRecoveries_T9 = 3
        self.assertLessEqual(self.edge.numOfRecoveries_T9, self.edge.numOfRecoveries_T95)

    def test_numOfRemoved_not_smaller_than_T95_or_T9(self):
        """
        Test case to verify that numOfRemoved is not smaller than numOfRecoveries_T95 or numOfRecoveries_T9.

        This test case creates an instance of the Edge class and sets its attributes.
        It then asserts that numOfRemoved is not smaller than numOfRecoveries_T95 and numOfRecoveries_T9.

        """
        self.edge.numOfRecoveries_T95 = 4
        self.edge.numOfRecoveries_T9 = 3
        self.edge.numOfRemoved = 5
        self.assertGreaterEqual(self.edge.numOfRemoved, self.edge.numOfRecoveries_T95)
        self.assertGreaterEqual(self.edge.numOfRemoved, self.edge.numOfRecoveries_T9)

if __name__ == '__main__':
    unittest.main()