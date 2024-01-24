# -*- coding: utf-8 -*-
"""

@author: VMoiseienko
"""
from task1 import count_islands 

# User input for matrix dimensions, cause otherwise not clear why provide dimention not just Matrix
rows, cols = map(int, input("Enter the number of rows and columns (separated by space): ").split())

# User input for the matrix
matrix = []
print("Enter the matrix values row by row:")
for i in range(rows):
    row = list(map(int, input(f"Row {i + 1} (space-separated values): ").split()))
    matrix.append(row)

# Calculate and print the number of islands
result = count_islands(matrix)
print("Number of islands:", result)