
## BP_RTO(Besov-Prior_Randomize-Then-Opimize): Code for Reproducing Results from (https://doi.org/10.1007/s11222-025-10638-2)

This repository contains the code and scripts necessary to reproduce the results presented in the preprint:  
**[Uncertainty Quantification for Linear Inverse Problems with Besov Prior: A Randomize-Then-Optimize Method]**

## Overview

The repository includes:
- **Jupyter Notebooks**: For visualizing and analyzing results, including CT reconstruction, inpainting, and deconvolution
- **Python Scripts**: Supporting modules for implementing algorithms like Besov priors and the Randomize-Then-Optimize method.
- **Data Handling**: Preloaded datasets for running experiments and reproducing plots.

## Structure
- `Sampling_Code/` : Contains python files with the implemented code needed for the numerical experiment and Jupyter Notebooks file
                     that that setups the numerical experiments, carries out the sampling, and saves the results for visualization.
- `Visualization_Code/`: Contains Jupyter Notebooks for generating figures and visualizations using data saved when running the Jupyter
                          Notebooks from Sampling_Code/.

## Requirements

- Python 3.10+
- Required libraries: `numpy`, `matplotlib`, `arviz`, `seaborn`, `pywt`, etc.
- Install dependencies using:
  ```bash
  pip install -r requirements.txt


## Usage

### Reproducing Figures

1. Clone the repository:
   ```bash
   git clone https://github.com/AndreasHorst/BP_RTO.git
   cd BP_RTO
   ```
2. Open the desired Jupyter Notebook in the `Visualization_Code/` folder (e.g., `CT_plot.ipynb`).
3. Run the notebook to generate the corresponding figures.

### Available Notebooks

- **`CT_plot.ipynb`**: Visualizations for CT reconstruction.
- **`Comparison_plot.ipynb`**: Comparison of sampling methods (RTO vs. NUTS).
- **`Inpainting_plot.ipynb`**: Analysis of inpainting techniques using Besov priors.
- **`Plot_Discretization_invariance.ipynb`**: Study of discretization invariance in Bayesian models.



Let me know if you'd like to add or modify anything!
