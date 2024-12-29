# Wafer Configurator

**Wafer Configurator** is a tool designed to generate optimal configurations for arranging KIDs (Kinetic Inductance Detectors) in an octagonal layout on a wafer. The goal is to minimize Crosstalk by maximizing the distance between neighboring KIDs' indices.

---

## Features

- **Generate Valid Configurations**: Create wafer layouts that meet constraints on minimum index differences between neighboring KIDs.
- **Evaluate Configurations**: Compute metrics such as cumulative distance, uniformity, radial balance, and grid smoothness to assess the quality of the configurations.
- **Visualize Results**: Create detailed plots, including individual configuration visualizations, radar charts, and heatmaps.


---

## Outputs

### Single KIDs Configuration
This plot shows the arrangement of KIDs in one configuration:

![Single Configuration Plot: diff ≥ 23](https://github.com/gpesce97/Thesis_Codes/blob/main/wafer_configurator/data/example_output/000_23_conf.png "Single KIDs Configuration")

---

### Overview of KIDs Configurations
The plot below represents the average index values across multiple configurations:

![Overview Plot](https://github.com/gpesce97/Thesis_Codes/blob/main/wafer_configurator/data/example_output/config_overview.png "Overview of KIDs Configurations")
