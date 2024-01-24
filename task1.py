# -*- coding: utf-8 -*-
"""

@author: VMoiseienko
"""
def count_islands(matrix):
    # Check if the matrix is empty
    if not matrix or not matrix[0]:
        return 0

    # Get the number of rows and columns in the matrix
    rows, cols = len(matrix), len(matrix[0])

    # Function to check if a cell is a valid part of an island
    def is_valid(i, j, visited):
        return 0 <= i < rows and 0 <= j < cols and not visited[i][j] and matrix[i][j] == 1

    # Depth-First Search (DFS) function to traverse the island
    def dfs(i, j, visited):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Possible movement directions
        visited[i][j] = True  # Mark the current cell as visited

        # Explore neighboring cells
        for dir_i, dir_j in directions:
            new_i, new_j = i + dir_i, j + dir_j
            if is_valid(new_i, new_j, visited):
                dfs(new_i, new_j, visited)

    visited = [[False] * cols for _ in range(rows)]  # Matrix to track visited cells
    island_count = 0  # Counter for the number of islands

    # Iterate through each cell in the matrix
    for i in range(rows):
        for j in range(cols):
            # If the cell is part of an unvisited island, initiate DFS
            if not visited[i][j] and matrix[i][j] == 1:
                island_count += 1
                dfs(i, j, visited)

    return island_count
