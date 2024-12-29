"""
Module: metrics.py
Author: Giacomo Lopez
Created on: May 28, 2024,
Version: 1.0.0
Last modified: December 29, 2024,
Description:
------------
This module provides a suite of metrics to evaluate the quality of KID (Kinetic Inductance Detector) configurations
on wafer layouts. These metrics enable the quantitative assessment of spatial and index-based distributions
to ensure minimal Crosstalk and optimal performance.

Purpose:
--------
The goal of this module is to provide tools for:
- Measuring distances and variabilities between neighboring KIDs.
- Evaluating the symmetry of KIDs' placement around the wafer center.
- Providing statistics that reflect the overall uniformity and separation of indices.

Key Features:
-------------
1. **Distance-Based Metrics**:
   - `get_cum_distance`: Computes the cumulative Euclidean distance between KIDs.
   - `distance_uniformity`: Measures the uniformity of distances between KIDs and their neighbors.
   - `max_neighbor_distance`: Determines the maximum distance between neighboring KIDs.

2. **Symmetry Analysis**:
   - `radial_balance`: Evaluates the balance of KIDs in symmetric positions around the wafer's center.

3. **Variability Metrics**:
   - `grid_smoothness`: Measures the smoothness of the grid based on differences between adjacent KIDs.
   - `distribution_stats`: Provides statistical measures (mean, median, std, entropy) of frequency differences.

Example Usage:
--------------
```python
import metrics as ms
import numpy as np

# Example wafer configuration
wafer = np.array([[0, 1, -8], [2, 3, 4], [-8, 5, 6]])
diff = 4

# Compute metrics
cum_distance = ms.get_cum_distance(wafer, diff)
uniformity = ms.distance_uniformity(wafer, diff)
radial_balance = ms.radial_balance(wafer, diff)

print("Cumulative Distance:", cum_distance)
print("Uniformity:", uniformity)
print("Radial Balance:", radial_balance)
"""

import numpy as np
from scipy.stats import entropy


def get_cum_distance(wafer: np.ndarray, diff: int) -> float:
    """
    Computes the cumulative Euclidean distance between KIDs on a wafer layout.

    This function calculates the sum of Euclidean distances between all pairs of
    KIDs in the wafer matrix, considering only KIDs that meet
    a certain condition based on their indices. The goal is to maximize this
    cumulative distance, as a higher value indicates better separation between KIDs.

    Parameters:
    - wafer (np.ndarray): A 2D array representing the wafer layout, where each cell
                          contains either a KID index or a mask value.
    - diff (int): The minimum difference between the indices of a KID.

    Returns:
    - float: The cumulative Euclidean distance between relevant pairs of KIDs.

    Metric Behavior:
    - A **higher value** of cumulative distance is desirable, as it indicates that
      KIDs are more evenly distributed across the wafer, resulting in better separation.

    Process:
    1. Precompute the positions of all KIDs in the wafer matrix.
    2. Iterate through all non-masked cells of the wafer.
    3. For each KID in the wafer, compute the Euclidean distance to a subset of
       neighboring KIDs (those with indices larger than the current KID).
    4. Sum up all distances to obtain the cumulative distance.

    Notes:
    - The mask value is calculated as `-2 * diff` and ignored in the distance calculation.
    - The neighbor KIDs considered are those with indices greater than the current KID,
      up to a maximum of 8 indices ahead (number of neighbor KIDs).

    Example:
    For a wafer layout:
        [[ 0,  1],
         [ 2, -4]],
    With `diff = 2`, the mask value is `-4`. The function computes the cumulative
    distances between valid KID positions.

    Raises:
    - KeyError: If any KID index is missing from the wafer matrix.

    """
    # Determine the total number of KIDs in the wafer by finding the maximum index + 1.
    N_KIDs = wafer.max() + 1
    mask_value = -2 * diff

    # Precompute the positions of all KIDs in the wafer for quick access.
    kid_positions = {kid: np.argwhere(wafer == kid)[0] for kid in range(N_KIDs)}

    cum_distance = 0

    for i in range(wafer.shape[0]):
        for j in range(wafer.shape[1]):
            # Skip cells that contain the mask value (invalid cells).
            if wafer[i, j] != mask_value:
                current_kid = wafer[i, j]

                # Iterate only over relevant neighbor KID indices (greater than current_kid).
                for neighbor_kid in range(current_kid + 1, min(current_kid + 8, N_KIDs)):
                    # Retrieve the position of the neighbor KID.
                    ni, nj = kid_positions[neighbor_kid]

                    # Compute the Euclidean distance between the current cell and the neighbor.
                    distance = ((ni - i) ** 2 + (nj - j) ** 2) ** 0.5
                    # Add the distance to the cumulative sum.
                    cum_distance += distance

    return cum_distance


def distance_uniformity(wafer: np.ndarray, diff: int) -> float:
    """
    Computes the uniformity of distance differences between neighboring KIDs in a wafer.

    This function calculates the standard deviation of the differences in indices
    between neighboring KIDs, normalized by the mean of those differences. A higher
    value indicates greater variability in distances, aligning with the goal of maximizing
    separation between KIDs and their neighbors.

    Parameters:
    - wafer (np.ndarray): A 2D array representing the wafer layout, where each cell
                          contains either a KID index or a mask value.
    - diff (int): The difference value used to compute the mask value.

    Returns:
    - float: The uniformity metric, calculated as the ratio of the standard deviation
             to the mean of the distance differences.

    Metric Behavior:
    - **Higher values are better**: A larger ratio indicates greater variability and
      higher average distances between neighboring KIDs, reflecting better separation.

    Process:
    1. Iterate over all cells in the wafer matrix.
    2. Skip masked cells (with value `-2 * diff`).
    3. For each valid cell, compute the absolute difference in indices with its
       8 immediate neighbors (if valid).
    4. Compute the mean and standard deviation of all collected differences.
    5. Return the ratio of the standard deviation to the mean.

    Example:
    For a wafer layout:
        [[ 0,  1],
         [ 2, -4]],
    With `diff = 2`, the mask value is `-4`. The function calculates the uniformity
    based on the differences between neighboring indices.

    Raises:
    - ZeroDivisionError: If the mean of the distances is zero.
    - ValueError: If no valid distances are found.
    """

    # Define the mask value, which represents invalid cells.
    mask_value = -2 * diff

    # Initialize a list to collect distances between neighboring KIDs.
    distances = []

    for i in range(wafer.shape[0]):
        for j in range(wafer.shape[1]):
            if wafer[i, j] != mask_value:
                # Iterate over all 8 neighboring cells (using relative offsets).
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, j + dj

                    # Check if the neighbor is within bounds.
                    if 0 <= ni < wafer.shape[0] and 0 <= nj < wafer.shape[1]:
                        neighbor_kid = wafer[ni, nj]

                        # Skip masked neighbors.
                        if neighbor_kid != mask_value:
                            distances.append(abs(wafer[i, j] - neighbor_kid))

    # Compute the mean and standard deviation of the collected distances.
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Return the uniformity metric (standard deviation normalized by the mean).
    return std_distance / mean_distance


def max_neighbor_distance(wafer: np.ndarray, diff: int) -> int:
    """
    Computes the maximum distance between a KID and its neighbors on a wafer.

    This function calculates the largest absolute difference in indices between
    any KID and its valid neighbors in the wafer matrix. The result indicates
    the maximum distance between adjacent KIDs.

    Parameters:
    - wafer (np.ndarray): A 2D array representing the wafer layout, where each cell
                          contains either a KID index or a mask value.
    - diff (int): The difference value used to compute the mask value.

    Returns:
    - int: The maximum absolute distance between a KID and its neighbors.

    Metric Behavior:
    - **Higher values are better**: A larger maximum distance indicates that there are
      significant separations between some adjacent KIDs

    Process:
    1. Iterate over all cells in the wafer matrix.
    2. Skip masked cells (with value `-2 * diff`).
    3. For each valid cell, compute the absolute difference in indices with its
       8 immediate neighbors (if valid).
    4. Track the largest distance encountered.

    Example:
    For a wafer layout:
        [[ 0,  1],
         [ 2, -4]],
    With `diff = 2`, the mask value is `-4`. The function calculates the largest
    absolute difference between neighboring indices.

    Raises:
    - None: Handles edge cases (e.g., empty wafer) gracefully by returning `0`.

    """
    # Define the mask value, which represents invalid cells.
    mask_value = -2 * diff
    max_distance = 0

    for i in range(wafer.shape[0]):
        for j in range(wafer.shape[1]):
            if wafer[i, j] != mask_value:
                # Iterate over all 8 neighboring cells (using relative offsets).
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    # Calculate the coordinates of the neighbor.
                    ni, nj = i + di, j + dj
                    if 0 <= ni < wafer.shape[0] and 0 <= nj < wafer.shape[1]:
                        neighbor_kid = wafer[ni, nj]

                        # Skip masked neighbors.
                        if neighbor_kid != mask_value:
                            # Update the maximum distance if the current distance is larger.
                            max_distance = max(max_distance, abs(wafer[i, j] - neighbor_kid))

    return max_distance


def distribution_stats(wafer: np.ndarray, diff: int) -> dict:
    """
    Computes statistical metrics for the distribution of frequency differences between neighboring KIDs.

    This function calculates the mean, median, standard deviation, and entropy of
    the absolute differences in indices between neighboring KIDs in a wafer layout.

    Parameters:
    - wafer (np.ndarray): A 2D array representing the wafer layout, where each cell
                          contains either a KID index or a mask value.
    - diff (int): The difference value used to compute the mask value.

    Returns:
    - dict: A dictionary containing the following statistics:
        - "mean": The average of the frequency differences.
        - "median": The median of the frequency differences.
        - "std": The standard deviation of the frequency differences.
        - "entropy": The entropy of the frequency difference distribution.

    Metric Behavior and Preferred Values:
    - **Mean**: Represents the average difference between neighboring KIDs.
      - **Higher values are preferable**, as they indicate that neighboring KIDs are more separated.
    - **Median**: Represents the midpoint of the distribution of differences.
      - **Higher values are preferable**, as they suggest that most neighboring KIDs have significant separation.
    - **Standard Deviation (std)**: Measures the spread of the differences.
      - **Lower values are preferable**, as they indicate more consistent and uniform differences between neighbors.
    - **Entropy**: Quantifies the randomness or variability in the differences.
      - **Higher values are preferable**, as they reflect greater diversity in neighboring distances

    Process:
    1. Iterate over all cells in the wafer matrix.
    2. Skip masked cells (with value `-2 * diff`).
    3. For each valid cell, compute the absolute difference in indices with its
       8 immediate neighbors (if valid).
    4. Collect all frequency differences into a list.
    5. Compute and return the required statistical metrics.

    Example:
    For a wafer layout:
        [[ 0,  1],
         [ 2, -4]],
    With `diff = 2`, the mask value is `-4`. The function calculates statistics
    based on the differences between neighboring indices.

    Raises:
    - ValueError: If no valid frequency differences are found.

    """
    # Define the mask value, which represents invalid cells.
    mask_value = -2 * diff
    freq_differences = []

    for i in range(wafer.shape[0]):
        for j in range(wafer.shape[1]):
            if wafer[i, j] != mask_value:
                # Iterate over all 8 neighboring cells (using relative offsets).
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, j + dj
                    # Check if the neighbor is within bounds.
                    if 0 <= ni < wafer.shape[0] and 0 <= nj < wafer.shape[1]:
                        neighbor_kid = wafer[ni, nj]
                        # Skip masked neighbors
                        if neighbor_kid != mask_value:
                            # Compute the absolute difference and append to the list.
                            freq_differences.append(abs(wafer[i, j] - neighbor_kid))

    stats = {
        "mean": np.mean(freq_differences),
        "median": np.median(freq_differences),
        "std": np.std(freq_differences),
        "entropy": entropy(freq_differences)
    }
    return stats


def radial_balance(wafer: np.ndarray, diff: int) -> float:
    """
    Computes the radial balance of KIDs in the wafer layout.

    This function calculates the sum of absolute differences between KIDs at
    symmetric positions with respect to the center of the wafer. A lower value
    indicates better radial balance, as it suggests more symmetry in the
    distribution of KIDs around the center.

    Parameters:
    - wafer (np.ndarray): A 2D array representing the wafer layout, where each cell
                          contains either a KID index or a mask value.
    - diff (int): The difference value used to compute the mask value.

    Returns:
    - float: The radial balance metric, calculated as the sum of absolute differences
             between KIDs at symmetric positions.

    Metric Behavior:
    - **Lower values are better**: A smaller radial difference indicates a more symmetric
      distribution of KIDs around the wafer's center.

    Process:
    1. Compute the center of the wafer.
    2. Iterate over all valid cells (non-masked).
    3. Identify the symmetric cell with respect to the center.
    4. If the symmetric cell is valid, compute the absolute difference between the
       KID indices of the two cells.
    5. Accumulate the differences to compute the radial balance.

    Example:
    For a wafer layout:
        [[ 0,  1],
         [ 2, -4]],
    With `diff = 2`, the mask value is `-4`. The function calculates the differences
    between symmetric KIDs (if valid) and sums them.

    Raises:
    - None: Handles edge cases like out-of-bound symmetric cells gracefully by skipping them.

    """
    # Define the mask value, which represents invalid cells.
    mask_value = -2 * diff

    # Determine the center coordinates of the wafer.
    center_x, center_y = wafer.shape[0] // 2, wafer.shape[1] // 2

    # Initialize the radial difference accumulator.
    radial_diff = 0

    for i in range(wafer.shape[0]):
        for j in range(wafer.shape[1]):
            # Skip masked cells.
            if wafer[i, j] != mask_value:
                # Compute the coordinates of the symmetric cell with respect to the center.
                opp_x, opp_y = 2 * center_x - i, 2 * center_y - j

                # Check if the symmetric cell is within bounds.
                if 0 <= opp_x < wafer.shape[0] and 0 <= opp_y < wafer.shape[1]:
                    # Skip if the symmetric cell is masked.
                    if wafer[opp_x, opp_y] != mask_value:
                        # Accumulate the absolute difference between the cell and its symmetric counterpart.
                        radial_diff += abs(wafer[i, j] - wafer[opp_x, opp_y])

    return radial_diff


def grid_smoothness(wafer: np.ndarray, diff: int) -> float:
    """
    Computes the grid smoothness of a wafer layout.

    This function calculates the average absolute difference between KIDs and their
    immediate neighbors in the wafer grid. A higher value indicates greater variability
    between neighboring KIDs, reflecting a less uniform grid with maximized differences.

    Parameters:
    - wafer (np.ndarray): A 2D array representing the wafer layout, where each cell
                          contains either a KID index or a mask value.
    - diff (int): The difference value used to compute the mask value.

    Returns:
    - float: The grid smoothness metric, calculated as the average absolute difference
             between neighboring KIDs.

    Metric Behavior:
    - **Higher values are better**: A larger smoothness value indicates greater differences
      between neighboring KIDs, which aligns with the goal of maximizing index variability.

    Process:
    1. Iterate over all valid cells (non-masked).
    2. For each valid cell, compute the absolute difference with its immediate neighbors
       (up, down, left, right).
    3. Accumulate the differences across all valid neighbors.
    4. Normalize the accumulated value by the total number of cells in the grid.

    Example:
    For a wafer layout:
        [[ 0,  1],
         [ 2, -4]],
    With `diff = 2`, the mask value is `-4`. The function calculates the smoothness
    based on the differences between valid neighboring indices.

    Raises:
    - ZeroDivisionError: If the wafer matrix has zero dimensions (handled implicitly by numpy).

    """
    # Define the mask value, which represents invalid cells.
    mask_value = -2 * diff

    # Initialize the smoothness accumulator.
    smoothness = 0

    # Iterate through all cells in the wafer matrix.
    for i in range(wafer.shape[0]):
        for j in range(wafer.shape[1]):
            # Skip masked cells.
            if wafer[i, j] != mask_value:
                current_kid = wafer[i, j]

                # Iterate over the four immediate neighbors (up, down, left, right).
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < wafer.shape[0] and 0 <= nj < wafer.shape[1]:
                        neighbor_kid = wafer[ni, nj]
                        # Skip masked neighbors.
                        if neighbor_kid != mask_value:
                            # Accumulate the absolute difference between the current KID and the neighbor.
                            smoothness += abs(current_kid - neighbor_kid)

    return smoothness / (wafer.shape[0] * wafer.shape[1])
