#!/bin/bash

python -m unittest discover -s tests -p 'test_*.py'

#### Additional options for running specific tests by terminal command:
# 1) python -m unittest tests.test_Edge [specific class test]
# 2) python -m unittest tests.test_Edge.TestEdge.test_edge_init_values [specific method test]

#### Additional options for running all tests by bash command:
# 1) python -m unittest discover -s tests -p 'test_*.py' [run all tests in the tests directory]
# 2) python -m unittest tests/test_Edge.py [run all tests in the Edge module




