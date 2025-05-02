```markdown
# BP_RTO: Code for Reproducing Results from [Research Square Preprint](https://www.researchsquare.com/article/rs-4528903/v1)

This repository contains the code and scripts necessary to reproduce the results presented in the preprint:  
**[Insert Paper Title Here]** ([Research Square](https://www.researchsquare.com/article/rs-4528903/v1)).

## Overview

The repository includes:
- **Jupyter Notebooks**: For visualizing and analyzing results, including CT reconstruction, inpainting, and discretization invariance.
- **Python Scripts**: Supporting modules for implementing algorithms like wavelet-based Besov priors and Bayesian deconvolution.
- **Data Handling**: Preloaded datasets for running experiments and reproducing plots.

## Structure

- `Visualization_Code/`: Contains Jupyter Notebooks for generating figures and visualizations.
- `Inpainting_Plot_Data/`, `Comparison_Plot_Data/`, etc.: Datasets used in the analysis.
- `blurring.py`: A utility module for simulating blurring operations.

## Requirements

- Python 3.10+
- Required libraries: `numpy`, `matplotlib`, `arviz`, `seaborn`, `pywt`, etc.
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

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
- **`Inpainting_plot.ipynb`**: Analysis of inpainting techniques using wavelets.
- **`Plot_Discretization_invariance.ipynb`**: Study of discretization invariance in Bayesian models.

## Citation

If you use this code or find it helpful, please cite the preprint:
```
[Insert Citation in Your Desired Format]
```

## License

This repository is licensed under the [MIT License](LICENSE).
```

Let me know if you'd like to add or modify anything!
