"""
Module: detector.py
Author: Giacomo Lopez
Created on: May 28, 2024,
Version: 1.0.0
Last modified: December 28, 2024

Description:
------------
This module generates configurations for arranging 145 KIDs (kinetic inductance
detectors) in an octagonal pattern on a wafer. Each KID is assigned a unique index,
and the arrangement ensures a minimum difference between the indices of a KID
and its neighbors (the 8 adjacent positions in the grid). This minimizes
Crosstalk caused by neighboring KIDs with close resonance frequencies.

Problem Statement:
-------------------
When arranging KIDs in a geometric pattern, Crosstalk can occur between neighboring
detectors if their resonance frequencies are too similar. This module generates
random configurations of the octagonal arrangement while enforcing a minimum
difference (`diff`) between the indices of adjacent KIDs.

Theoretical Background:
------------------------
This problem is rooted in graph theory and is closely related to Graph Labeling
and Channel Assignment problems. In mathematical terms, the goal can be described as:

    "Label each vertex of a grid-graph with a distinct integer so that adjacent
    vertices differ by at least `diff`."

Such problems are well-known in the field of Graph Labeling, where the goal is to assign
labels (often integers) to vertices of a graph while satisfying specific constraints.
A common application is the Channel Assignment problem in telecommunications, where
channels (frequencies) must be assigned to nodes (transmitters) such that interference
is minimized.

Complexity:
-----------
For large grids (N), this type of problem is generally NP-hard. This means there is
currently no known closed-form formula that provides a general solution for all grid
sizes and all `diff` values. Specifically, there is no formula of the type:

    diff_max = f(N)

that guarantees a valid configuration for arbitrary N. Instead, solutions often rely
on heuristic or computational approaches, as implemented in this module.


Solution:
---------
The algorithm:
1. Defines an octagonal mask within a 13x13 grid.
2. Randomly assigns KIDs to the valid positions within the octagon.
3. Ensures that no two adjacent positions have indices differing by less than `diff`.
"""
import os
import pandas as pd

import numpy as np
import multiprocessing as mp

import shutil
import seaborn as sns

import metrics as ms
import radar_chart as rc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Agg is an engine for rendering static visualizations used for saving
# images to files rather than displaying them interactively
matplotlib.use('Agg')

# if latex is installed, it will be used to format the plots
if shutil.which('latex'):
    plt.rcParams['text.usetex'] = True


class WaferConfigurator:
    """
    Class: WaferConfigurator
    ----------------
    The `WaferConfigurator` class provides tools for generating and visualizing configurations of 145 KIDs
    (kinetic inductance detectors) arranged in an octagonal pattern on a wafer. It ensures a minimum
    difference between indices of neighboring KIDs to minimize Crosstalk caused by resonance frequency proximity.

    Key Features:
    - Constructs an octagonal wafer layout within a 13x13 matrix.
    - Generates random configurations of KID indices, enforcing minimum difference constraints.
    - Computes a variety of metrics to evaluate configurations.
    - Visualizes individual configurations and comparative summaries (e.g., radar charts, heatmaps).

    Attributes:
    ------------
    - `diff` (int): Minimum difference between indices of adjacent KIDs.
    - `wafer` (np.ndarray): The current wafer layout matrix.
    - `configurations` (list[np.ndarray]): List of generated wafer configurations.
    - `output` (str): Directory for saving visualization output.
    - `N` (int): Size of the wafer matrix (13x13 grid).
    - `n_KIDs` (int): Total number of KIDs to be distributed (145).
    - `metrics` (dict): Metrics computed for each configuration.

    Methods:
    --------
    1. `configure()`: Configures the output directory.
    2. `get_wafer()`: Constructs an octagonal wafer layout.
    3. `get_conf()`: Generates a single valid KID configuration.
    4. `get_confs(n_confs)`: Generates multiple configurations in parallel.
    5. `get_metrics()`: Computes evaluation metrics for each configuration.
    6. `plot_conf(ax, conf)`: Visualizes a configuration as a grid.
    7. `plot_stats(ax, metric)`: Displays metrics as a table.
    8. `radar_chart(ax, df)`: Creates a radar chart comparing configurations.
    9. `heatmap(ax, df)`: Creates a heatmap of normalized configuration metrics.
    10. `overview(metrics, output)`: Generates an overview of configurations (radar + heatmap).
    11. `plot_results(id_conf, conf, metric, diff, output)`: Visualizes a single configuration.
    12. `plot_confs(offset)`: Visualizes all configurations in parallel.
    """

    def __init__(self, diff: int, output: str):
        self.diff: int = diff
        self.wafer: np.ndarray = np.array([])
        self.configurations: list[np.ndarray] = []
        self.output: str = output

        # order of the matrix that will contain the octagon
        self.N = 13

        # number of kids to be distributed in the configuration
        self.n_KIDs = 145

        # dict containing all the metrics about each configuration found
        self.metrics: dict = {}

    def configure(self):
        """
        Configures the output directory for the current process.

        This method updates the `self.output` attribute by appending a unique timestamp
        (based on the elapsed process time) to the existing output path. It ensures that
        the directory is created if it does not already exist.

        Steps:
            1. Combines the current `self.output` path with a rounded value of the elapsed
               process time to create a unique directory name.
            2. Creates the directory specified by the updated `self.output` path, including
               any necessary parent directories, without raising an error if it already exists.

        Notes:
            - The `os.times().elapsed` value represents the process's elapsed time in seconds.
            - The `round` function is used to produce an integer timestamp for simplicity and
              compatibility with filesystem naming conventions.

        Raises:
            - Any exceptions related to file path creation (e.g., permissions issues)
              are propagated.
        """

        self.output = os.path.join(self.output, str(round(os.times().elapsed)))
        os.makedirs(self.output, exist_ok=True)

    def get_wafer(self):
        """
        Generates and configures a square matrix to represent a wafer layout with an octagonal mask.

        This method creates a 2D numpy array (`self.wafer`) of size `self.N x self.N`,
        initializes it with a default fill value, and applies an octagonal mask to the corners
        by setting specific cells to a `mask_value`. The resulting matrix can be used to represent
        a wafer layout.

        Key Steps:
        1. **Initialize Matrix**:
           - The matrix (`self.wafer`) is initialized with the `fill_value`, which is defined as
             `self.n_KIDs + self.diff`.
        2. **Mask Corners**:
           - The corners of the matrix are masked to form an octagonal shape.
           - The mask is applied to the first and last 3 rows and columns, offset by specific logic.
           - Masked cells are set to `mask_value`, defined as `-2 * self.diff`.

        Parameters:
        - `self.N` (int): The size of the square matrix.
        - `self.diff` (int): A value used to calculate the `mask_value`.
        - `self.n_KIDs` (int): A value used to calculate the `fill_value`.

        Attributes Set:
        - `self.wafer` (np.ndarray): A square matrix of size `self.N x self.N` containing:
            - Masked cells (set to `mask_value`).
            - Non-masked cells (initialized to `fill_value`).

        Notes:
        - The offset logic ensures symmetry and positions the masked areas correctly to form an octagon.
        - The masked cells in the corners are adjusted using both the `x` and `y` loop indices.

        Example:
        For `self.N = 13`, `self.diff = 4`, `self.n_KIDs = 145`:
        - `fill_value = 149` (`self.n_KIDs + self.diff`)
        - `mask_value = -8` (`-2 * self.diff`)
        - Resulting matrix will have:
            - Cells in the first and last 3 rows/columns partially masked.
            - The rest initialized with `fill_value`.

        Raises:
        - Indexing errors if the mask logic is incompatible with the matrix dimensions.
        """

        # mask the corners of the square matrix to obtain an octagon
        mask_value = -2 * self.diff
        # specific offset of the octagon to be obtained
        offset = 10

        # the cells in which to place the indices of each kid are marked with the ‘fill_value’ value
        fill_value = self.n_KIDs + self.diff

        # get a starting configuration
        self.wafer = np.full(shape=(self.N, self.N), dtype=int, fill_value=fill_value)

        # the values to be masked to obtain the octagon are present in the first/last 3 rows and the first/last 3
        # columns of each row
        for x in range(3):
            for y in range(3 - x):
                # place the invalid cells at -diff, so that the sum with diff generates 0
                self.wafer[x, y] = self.wafer[y + offset + x, x] = self.wafer[x, y + x + offset] = self.wafer[
                    self.N - x - 1, self.N - y - 1] = mask_value

    def get_conf(self, _: None):
        """
        Generates a random configuration of kids indices on the wafer matrix.

        This method fills the octagonal region of the wafer matrix (`self.wafer`) with
        indices of kids, ensuring that a specific minimum difference (`self.diff`)
        is maintained between neighboring indices. The function uses a random number
        generator for assigning indices and validates configurations to enforce constraints.

        Parameters:
        - `_` (None): Placeholder parameter, not used in the function.

        Process:
        1. **Initialize Random Generator**:
           - Uses the system entropy for randomness via `np.random.default_rng()`
             without explicit seeding.
        2. **Define Fill Value**:
           - Cells eligible for filling have an initial value of `self.n_KIDs + self.diff`.
        3. **Generate Kids Indices**:
           - A list of indices (`kids`) ranging from `0` to `self.n_KIDs - 1` is created.
        4. **Iterate Over Octagonal Cells**:
           - For each eligible cell in the octagonal region of `self.wafer`:
             - Extract a `1x1` neighborhood of adjacent cells.
             - Randomly select a kid index that satisfies the minimum distance constraint
               (`self.diff`) relative to all neighboring values in the window.
             - If no valid kid can be assigned, the function returns early, indicating
               an invalid configuration.
        5. **Assign Kids**:
           - Once a valid kid is selected for a cell, it is placed in the matrix, and
             removed from the list of available kids.

        Returns:
        - `np.ndarray`: The updated `self.wafer` matrix with a valid configuration of
          kids indices if successful. If no valid configuration can be created, the
          function returns `None`.

        Constraints:
        - Cells are assigned only if their initial value matches `fill_value`.
        - Each kid index can be used only once.
        - Neighboring cells must have a difference of at least `self.diff` from the
          assigned value.

        Notes:
        - This function is designed to run in parallel processes for maximizing the
          search for valid configurations.

        Example:
        For `self.N = 13`, `self.n_KIDs = 145`, `self.diff = 4`:
        - The `self.wafer` matrix is initialized with eligible cells marked as
          `self.n_KIDs + self.diff = 149`.
        - Each valid configuration ensures neighboring cells differ by at least 2.

        Raises:
        - No explicit exceptions are raised, but the function may return early with `None`
          if a valid configuration cannot be achieved.
        """

        # initialize random generator.
        # since this function is executed in parallel to maximise the search for a configuration
        # with a specific difference, the random number generator is not initialized, but the entropy
        # of the system is used

        rng = np.random.default_rng()

        # the cells in which to place the indices of each kid are marked with the ‘fill_value’ value
        fill_value = self.n_KIDs + self.diff

        # generates list of kids indices
        kids = list(range(self.n_KIDs))

        wafer = self.wafer.copy()

        for i in range(self.N):
            for j in range(self.N):

                # I only consider cells belonging to the octagon
                # i.e. those that have `fill_value` as their initial value.
                if wafer[i][j] == fill_value:

                    # consider the 1-wide window that includes all the neighbours of a kid in the 8 adjacent positions
                    sub = wafer[max(0, i - 1): min(i + 2, len(wafer)),
                          max(j - 1, 0): min(j + 2, wafer.shape[1])]

                    # keeps track of extracted kids that do not fulfil the condition on minimum distance
                    # to neighbours in the window
                    tried_kids = set()

                    while True:
                        # Check if all kids have been tested
                        if len(tried_kids) == len(kids):
                            # I have tested all the available kids and none fulfils the minimum difference condition.
                            # The configuration is invalid
                            return None

                        # extract a random kid
                        kid = rng.choice(kids)

                        # if kid has already been tested, skip it
                        if kid in tried_kids:
                            continue

                        # add the kid extract to those tested
                        tried_kids.add(kid)

                        # check the distance with the KIDs actually placed
                        if np.all(np.abs(sub - kid) >= self.diff):

                            if kid in wafer:
                                raise ValueError(f"KID {kid} è già stato piazzato, ma viene ripiazzato!")

                            wafer[i][j] = kid
                            kids.remove(kid)
                            break

        return wafer

    def get_confs(self, n_confs: int):
        """
        Generates multiple configurations of the wafer matrix in parallel processes.

        This method spawns parallel processes to search for valid configurations of
        the wafer matrix. The results are filtered to retain valid configurations,
        sorted by a specified metric, and the top configurations are stored.

        Parameters:
        - `n_confs` (int): The number of configurations to generate in parallel.

        Process:
        1. **Parallel Processing**:
           - A multiprocessing pool (`mp.Pool`) is created, where `n_confs` parallel processes
             are instantiated. Each process calls the `get_conf` function to generate a configuration.
        2. **Filter Valid Configurations**:
           - Results from `get_conf` are filtered to include only valid configurations
             (non-empty arrays with valid entries).
        3. **Sort Configurations**:
           - Valid configurations are sorted in descending order based on the cumulative
             distance metric, calculated using `ms.get_cum_distance`.
        4. **Select Top Configurations**:
           - The top 5 configurations (based on the cumulative distance metric) are retained
             and stored in `self.configurations`.

        Attributes Set:
        - `self.configurations` (list of np.ndarray): A list of the top 5 configurations,
          sorted by their cumulative distance values.

        Notes:
        - The `get_conf` method is called within each parallel process, enabling faster
          generation of configurations.
        - The sorting ensures that the configurations with the highest cumulative distances
          are prioritized.

        Returns:
        - None: The method updates the `self.configurations` attribute directly.

        Example:
        For `n_confs = 10`, the method:
        - Spawns 10 parallel processes.
        - Filters out invalid configurations (if any).
        - Sorts the remaining valid configurations by cumulative distance.
        - Retains the top 5 configurations.

        Raises:
        - Any exceptions related to multiprocessing or the `get_conf` method will propagate.

        Dependencies:
        - The `metrics.get_cum_distance` method is used for sorting configurations.

        """

        with mp.Pool() as pool:
            results = pool.map(self.get_conf, range(n_confs))
            self.configurations = [res for res in results if np.any(res)]

            self.configurations = sorted(self.configurations,
                                         key=lambda x: ms.get_cum_distance(x, self.diff),
                                         reverse=True)[:5]

        if not self.configurations:
            raise Exception("No valid configurations could be achieved.")

    def get_metrics(self):
        """
        Computes a set of metrics for each configuration and stores the results.

        This method iterates over the `self.configurations` attribute and calculates
        various metrics for each configuration using the `metrics` module functions. The
        calculated metrics are stored in the `self.metrics` dictionary, with the
        configuration index as the key.

        Metrics computed for each configuration:
        1. **Cumulative Distance**: Total cumulative distance based on the configuration
           and the provided `self.diff`.
        2. **Uniformity**: A measure of the uniformity of distances.
        3. **Max Distance**: The maximum distance between neighboring points.
        4. **Radial Balance**: A metric evaluating balance in radial arrangements.
        5. **Grid Smoothness**: A measure of the smoothness of the grid layout.
        6. **Mean Frequency Diff**: The mean value of frequency differences.
        7. **Median Frequency Diff**: The median value of frequency differences.

        Parameters:
        - `self.configurations`: A list of configurations to process.
        - `self.diff`: The differences used for metric computation.

        Returns:
        - None: The metrics are stored in the `self.metrics` dictionary.

        Notes:
        - The `ms` module functions are used to compute the metrics.
        - The results are indexed starting from 1 to align with human-readable indexing.

        Raises:
        - Any exceptions raised by the `ms` module functions will propagate through this method.

        Example:
        If `self.configurations` contains 3 configurations, the resulting `self.metrics`
        dictionary will have the structure:
            {
                1: {"Cumulative Distance": ..., "Uniformity": ..., ...},
                2: {"Cumulative Distance": ..., "Uniformity": ..., ...},
                3: {"Cumulative Distance": ..., "Uniformity": ..., ...},
            }
        """

        for idx, conf in enumerate(self.configurations, start=1):
            metrics = {
                "Cumulative Distance": ms.get_cum_distance(conf, self.diff),
                "Uniformity": ms.distance_uniformity(conf, self.diff),
                "Max Distance": ms.max_neighbor_distance(conf, self.diff),
                "Radial Balance": ms.radial_balance(conf, self.diff),
                "Grid Smoothness": ms.grid_smoothness(conf, self.diff),
                "Mean Frequency Diff": ms.distribution_stats(conf, self.diff)["mean"],
                "Median Frequency Diff": ms.distribution_stats(conf, self.diff)["median"],
                "Entropy":  ms.distribution_stats(conf, self.diff)["entropy"],
            }

            self.metrics[idx] = metrics

    def plot_conf(self, ax: plt.Axes, conf: np.ndarray):
        """
        Visualizes a configuration on a given matplotlib axis.

        This method plots a 2D configuration grid (`conf`) on the provided matplotlib
        `ax` object, displaying each non-masked value as a formatted label in the grid.
        The visualization highlights each value with a styled bounding box.

        Parameters:
        - ax (plt.Axes): The matplotlib Axes object on which the configuration will be plotted.
        - conf (np.ndarray): A 2D numpy array representing the configuration to be visualized.

        Key Functionality:
        - **Masked Values**: Cells with a value equal to `-2 * self.diff` are treated as masked
          and are not visualized.
        - **Bounding Boxes**: Each non-masked value is displayed in a round-edged box with
          an orange border and white background.
        - **Text Formatting**: Values are incremented by 1, zero-padded to three digits,
          and displayed with LaTeX math formatting (`$...$`).
        - **Text Placement**: Each value is placed within its corresponding grid cell,
          normalized to the axis dimensions.

        Axis Configuration:
        - The axis background is set to white.
        - The aspect ratio is set to "equal" to ensure a uniform grid appearance.
        - The axis borders and labels are hidden for a cleaner look.

        Raises:
        - Any exceptions related to invalid Axes operations or grid shapes will propagate.

        Example:
        If `conf` is:
            [[ 0,  1],
             [ -4, -2 * self.diff]],
        The method will render values "001" and "002" in the first row with bounding boxes,
        skipping the masked value in the second row.
        """

        mask_value = -2 * self.diff

        for x in range(conf.shape[0]):
            for y in range(conf.shape[1]):

                if conf[x, y] != mask_value:
                    bbox = dict(boxstyle=patches.BoxStyle("Round", pad=0.4),
                                edgecolor='orange',
                                facecolor='white')

                    ax.text(**dict(x=x / conf.shape[0], y=y / conf.shape[1],
                                   s=f"${conf[x][y] + 1:03}$", size=9, rotation='horizontal', bbox=bbox))

        ax.set(facecolor='white', label=False, aspect='equal')
        ax.set_axis_off()

    @staticmethod
    def plot_stats(ax: plt.Axes, metric: dict):

        metric_names = list(metric.keys())
        metric_values = [rf"${value:.2f}$" for value in metric.values()]
        table_data = [[rf"${name.replace(' ', '~')}$", value] for name, value in zip(metric_names, metric_values)]

        # Crea e aggiungi la tabella
        table = ax.table(
            cellText=table_data,
            colLabels=["$ \\bf{Metric}$", "$ \\bf{Value}$"],
            loc="bottom",
            cellLoc="center",
            bbox=[0.3, 0.3, 0.5, 0.5]  # x, y, width, height in frazioni dell'area dell'asse
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(metric))))
        # table.auto_set_row_height(True)
        ax.set_aspect('equal')
        ax.set_axis_off()

    @staticmethod
    def radar_chart(ax: plt.Axes, df: pd.DataFrame):

        # retrieve all the metrics with arrows
        spoke_labels = list(map(lambda x: fr"${x.replace('↑', r'\uparrow')
                                .replace('↓', r'\downarrow')
                                .replace(' ', r'~')}$", df.index))
        # retrieve all the configuration names
        legend_labels = list(map(lambda x: rf"${x.split(' ')[1]}$", df.columns))
        # number of angles for radar chart
        N = len(spoke_labels)
        theta = rc.radar_factory(N, frame='polygon')
        # it is used to configure radial grids in a radar plot (or polar plot)
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])

        for i in range(len(df.columns)):
            ax.plot(theta, df.iloc[:, i])
            ax.fill(theta, df.iloc[:, i], alpha=0.25, label='_nolegend_')

        # Assegna le etichette e raccogli i Text objects

        _, tick_labels = ax.set_varlabels(spoke_labels)  # ax.set_thetagrids(angles_deg, labels)

        # si puo' ottimizzare definendola in base agli angoli contenuti in theta (rad)
        alignments = ['center', 'right', 'right', 'right', 'center', 'left', 'left', 'left']
        for label, align in zip(tick_labels, alignments):
            label.set_horizontalalignment(align)

        ax.legend(legend_labels, title=r"$Configurations$", loc=(1.2, .95), labelspacing=0.1, fontsize='small')

    @staticmethod
    def heatmap(ax: plt.Axes, df: pd.DataFrame):

        yticks_labels = list(map(lambda x: fr"${x.replace('↑', r'\uparrow')
                                 .replace('↓', r'\downarrow')
                                 .replace(' ', '~')}$", df.index))
        xticks_labels = list(map(lambda x: rf"${x.split(' ')[1]}$", df.columns))

        sns.heatmap(df,
                    annot=True,
                    cmap="coolwarm",
                    yticklabels=yticks_labels,
                    xticklabels=xticks_labels,
                    cbar_kws={'label': '$Normalized~Score$'},
                    fmt=".2f",
                    ax=ax)

    @staticmethod
    def overview(metrics: dict, output: str):
        """
        Generates a visual overview of wafer configurations based on given metrics.

        This method creates a combined radar chart and heatmap visualization to
        qualitatively and quantitatively compare multiple wafer configurations
        based on their computed metrics. The resulting visualization is saved as
        an image file.

        Parameters:
        - metrics (dict): A dictionary where each key is a configuration index (e.g., `1`, `2`, ...)
                          and each value is a dictionary of metrics for that configuration.
                          Example:
                          {
                              1: {"Metric A": value, "Metric B": value, ...},
                              2: {"Metric A": value, "Metric B": value, ...},
                              ...
                          }
        - output (str): Path to the directory where the visualization image (`config_overview.png`)
                        will be saved.

        Process:
        1. **Prepare Data**:
           - Convert the `metrics` dictionary into a DataFrame for easier manipulation.
           - Append arrows (`↑` for higher-is-better, `↓` for lower-is-better) to metric names
             based on the `higher_is_better` list.
           - Normalize metric values for each configuration:
             - Scale higher-is-better metrics such that the best value is 1 and the worst is 0.
             - Scale lower-is-better metrics inversely.
        2. **Create Visualizations**:
           - Generate a radar chart to display the qualitative distribution of metrics across
             configurations.
           - Generate a heatmap to provide a quantitative overview of metric values.
        3. **Save Output**:
           - Combine the radar chart and heatmap into a single figure.
           - Save the figure as `config_overview.png` in the specified output directory.

        Visualizations:
        - **Radar Chart**:
          - Plots each configuration on a radar chart using the normalized metric values.
          - Displays metric names (with arrows) as axes. The arrows indicate whether a high or low value is preferable
        - **Heatmap**:
          - Shows a color-coded matrix of normalized metric values.
          - Provides an easy-to-read comparison of all configurations.

        Returns:
        - None: The visualization is saved as an image in the output directory.

        Dependencies:
        - Requires the `WaferConfigurator.radar_chart` and `WaferConfigurator.heatmap` methods to generate
          the radar chart and heatmap, respectively.
        - Requires `matplotlib`, `pandas`, and `numpy`.

        Example:
        For a metrics dictionary:
        ```
        {
            1: {"Cumulative Distance": 100, "Uniformity": 0.8, "Max Distance": 15},
            2: {"Cumulative Distance": 90, "Uniformity": 0.85, "Max Distance": 12},
        }
        ```
        The resulting `config_overview.png` will contain:
        - A radar chart comparing the metrics of configuration 1 and 2.
        - A heatmap visualizing the normalized values of the metrics.

        Raises:
        - Any exceptions related to file I/O will propagate when saving the figure.
        """

        # Prepare data dictionary with configuration names as keys.
        # Each "Config X" corresponds to a set of metrics for that configuration.
        data_dict = {f"Config {config}": values for config, values in metrics.items()}

        # Define whether higher values are better (↑) or lower values are better (↓) for each metric.
        higher_is_better = [True, True, True, False, True, True, True, True]  # True = "↑", False = "↓"

        # Extract the metric names from the first configuration's metrics dictionary.
        metric_names = list(metrics[1].keys())

        # Add arrows (↑ or ↓) to metric names based on the higher_is_better list.
        metrics_with_arrows = [
            f"{metric} {'↑' if is_higher else '↓'}" for metric, is_higher in zip(metric_names, higher_is_better)
        ]

        # Create a DataFrame from the data dictionary with configurations as columns and metrics as rows.
        df = pd.DataFrame.from_dict(data_dict, orient="columns")
        # Set metric names (with arrows) as the row indices for better readability.
        df.index = metrics_with_arrows

        # Normalize the metrics to ensure comparability across different configurations.
        for i, better in enumerate(higher_is_better):
            if df.iloc[i].min() == df.iloc[i].max():
                # If all values for a metric are the same, set them to 0 (no meaningful comparison).
                df.iloc[i] = 0
            elif better:
                # For higher-is-better metrics, normalize so the minimum is 0 and the maximum is 1.
                df.iloc[i] = (df.iloc[i] - df.iloc[i].min()) / (df.iloc[i].max() - df.iloc[i].min())
            else:
                # For lower-is-better metrics, normalize inversely (minimum is 1, maximum is 0).
                df.iloc[i] = (df.iloc[i].max() - df.iloc[i]) / (df.iloc[i].max() - df.iloc[i].min())

        fig = plt.figure(figsize=(10, 8), layout="constrained")
        fig.suptitle("$Wafer Configurations: Qualitative and Quantitative Visualization$".replace(' ', '~'),
                     fontsize=12)

        # Register the radar projection for the radar chart.
        _ = rc.radar_factory(len(df.index), frame='polygon')

        # Create a subplot for the radar chart in the upper half of the figure.
        ax_radar = fig.add_subplot(2, 1, 1, projection="radar")
        ax_radar.set_aspect('equal', anchor='C')

        # Create another subplot for the heatmap in the lower half of the figure.
        ax_heatmap = fig.add_subplot(2, 1, 2)

        # Generate the heatmap visualization
        WaferConfigurator.heatmap(ax_heatmap, df)

        # Generate the radar chart visualization
        WaferConfigurator.radar_chart(ax_radar, df)

        fig.savefig(os.path.join(output, 'config_overview.png'), dpi=600)

    def plot_results(self, id_conf: int, conf: np.ndarray, metric: dict, diff: int, output: str):
        """
        Generates and saves a visualization of a specific wafer configuration and its associated metrics.

        This method creates a two-panel figure:
        1. A graphical representation of the wafer configuration.
        2. A tabular summary of the associated metrics.
        The figure is saved as a high-resolution PNG file in the specified output directory.

        Parameters:
        - id_conf (int): Identifier for the configuration being plotted (e.g., configuration number).
        - conf (np.ndarray): A 2D array representing the wafer configuration.
        - metric (dict): A dictionary containing metrics for the configuration.
        - diff (int): The difference value associated with the configuration.
        - output (str): Path to the directory where the visualization file will be saved.

        Process:
        1. **Figure Setup**:
           - Creates a figure with two subplots arranged side-by-side.
           - Adds a title to the figure, formatted with the configuration ID.
        2. **Plot Configuration**:
           - Uses `self.plot_conf` to render the wafer configuration on the first subplot.
        3. **Plot Metrics**:
           - Uses `self.plot_stats` to display the metrics in a table format on the second subplot.
        4. **Save Output**:
           - Saves the figure as a PNG file with a name containing the configuration ID and diff value.

        Returns:
        - None: The visualization is saved to the output directory.

        Example:
        For `id_conf = 1`, `diff = 3`, the output file would be named `001_3_conf.png`.

        Raises:
        - Any exceptions related to file saving or plotting will propagate.

        """

        fig, (ax_conf, ax_table) = plt.subplots(1, 2, figsize=(10, 8), tight_layout=True)
        fig.suptitle(rf'$ Wafer Configuration \#{id_conf:03}: diff \ge {diff}$'.replace(' ', '~'), fontsize=14, fontweight='bold')

        self.plot_conf(ax_conf, conf)
        self.plot_stats(ax_table, metric)

        fig.savefig(os.path.join(output, f'{id_conf:03}_{diff}_conf.png'), dpi=600)

    def plot_confs(self, offset: int):
        """
        Generates visualizations for multiple wafer configurations and saves them as images.

        This method performs the following steps:
        1. Computes metrics for all configurations.
        2. Creates an overview visualization of the configurations and their metrics.
        3. Uses multiprocessing to generate individual plots for each configuration,
           with metrics and layout visualizations saved as separate files.

        Parameters:
        - offset (int): The starting index for configuration IDs in the plots. This is useful
                        when generating plots in batches to avoid overlapping IDs.

        Process:
        1. **Metrics Calculation**:
           - Calls `self.get_metrics()` to compute metrics for all configurations.
        2. **Overview Visualization**:
           - Calls `self.overview()` to create a combined radar chart and heatmap visualization.
        3. **Parallel Plotting**:
           - Uses a multiprocessing pool to generate plots in parallel:
             - Each process calls `self.plot_results()` with the configuration data.
             - Configuration IDs start at `offset`.
             - Metrics and configurations are passed in batches, and the workload is evenly distributed
               across CPUs using chunks.

        Attributes Used:
        - `self.configurations` (list): The list of configurations to plot.
        - `self.metrics` (dict): The computed metrics for each configuration.
        - `self.diff` (int): The minimum difference value used in metrics and filenames.
        - `self.output` (str): Directory where the output files are saved.

        Returns:
        - None: All plots are saved as PNG files in the output directory.

        Example:
        If `offset=10` and there are 5 configurations:
        - Configuration IDs in the plots will be 10, 11, 12, 13, and 14.

        Raises:
        - Any exceptions related to multiprocessing, plotting, or file I/O will propagate.

        """

        # Step 1: Compute metrics for all configurations.
        self.get_metrics()

        # Step 2: Create an overview visualization of the metrics.
        self.overview(self.metrics, self.output)

        # Step 3: Set up multiprocessing for parallel plotting.
        with mp.Pool() as pool:
            # Number of configurations to process.
            n_confs = len(self.configurations)

            _ = pool.starmap(self.plot_results,
                             zip(range(offset, n_confs + offset),   # Configuration IDs starting from `offset`.
                                 self.configurations,               # List of configurations to plot.
                                 self.metrics.values(),             # Corresponding metrics for each configuration.
                                 n_confs * [self.diff],             # Repeated `diff` value for all configurations.
                                 n_confs * [self.output]),          # Repeated output directory for all plots.
                             chunksize=max(1, n_confs // mp.cpu_count()))  # Determine chunk size based on CPU count.


if __name__ == '__main__':
    wConf = WaferConfigurator(diff=23, output='test_with_class')
    wConf.configure()
    wConf.get_wafer()
    wConf.get_confs(100000)
    wConf.get_metrics()
    wConf.plot_confs(0)
