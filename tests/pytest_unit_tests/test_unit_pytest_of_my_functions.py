import math
from math import exp
import numpy  as np
import pytest

# Function to be tested
def my_softmax_function(input_vector):
    # Calculate the exponent of each element in the input vector
    exponents = [exp(j) for j in input_vector]
    # divide the exponent of each value by the sum of the
    # exponents and round of to 3 decimal places
    # probas = [round(exp(i) / sum(exponents), 1) for i in input_vector]
    probas = [exp(i) / sum(exponents) for i in input_vector]

    return probas


# Actual pytest testing function
def test_my_softmax_function():

  # Arrange step (i.e. setup what to test)
  input_value = [0, np.log(2), np.log(7)]
  expected_result = [0.1, 0.2, 0.7]

  # Act step
  outcome = my_softmax_function(input_value)

  # Assert step
  assert outcome == expected_result

