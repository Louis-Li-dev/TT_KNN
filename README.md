
# Predicting Human Mobility Patterns Using the Temporal-Trajectory-based K Nearest Neighbor Algorithm

![License](https://img.shields.io/badge/License-MIT-green.svg)  
[![ACM Paper DOI Badge](https://img.shields.io/badge/ACM%20Paper-DOI%3A%2010.1145%2F3681771.3699913-blue)](https://dl.acm.org/doi/10.1145/3681771.3699913)

## Architecture

![Architecture Diagram](https://github.com/user-attachments/assets/d2261ac1-1739-410b-b500-c2ed82ede9fa)

## Dataset

You can download the dataset used in this project from Zenodo:  
[📥 Download Dataset from Zenodo](https://zenodo.org/records/14219563)

## Additional Packages and Setup

To run the evaluation metric, you'll need to clone the `geobleu` repository. Follow these steps to get everything set up:

### Clone the Evaluation Metric Repository

Clone the repository for **Geobleu**, an evaluation metric used in this project:  
[🌐 Geobleu GitHub Repository](https://github.com/yahoojapan/geobleu)

## How to Run

Import the `ttknn` directory from this repository and use the `temporal_knn_fit_predict` function.  
The [`example.ipynb`](https://github.com/Louis-Li-dev/TT_KNN/blob/main/tests/example.ipynb) notebook demonstrates how to import and utilize the `ttknn` directory effectively.

## Installation

To install all the necessary dependencies, run the following command:

```bash
git clone https://github.com/Louis-Li-dev/TT_KNN
pip install -r requirements.txt
```

or

```bash
pip install git+https://github.com/Louis-Li-dev/TT_KNN
```
## Testing and Experiment
In directory `tests`,
```bash
  python run_experiment.py --user_range 3000 --data_folder ../cityB-dataset --result_folder ../result
```
## Citation
```bibtex
@inproceedings{10.1145/3681771.3699913,
  author = {Li, An-Syu and Meng, Ling-Huan and Zhong, Yu-Ling and Chen, Yi-Chung and Kawakami, Tomoya},
  title = {Using the Temporal-Trajectory-based K Nearest Neighbor Algorithm to Predict Human Mobility Patterns},
  year = {2024},
  isbn = {9798400711503},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3681771.3699913},
  doi = {10.1145/3681771.3699913},
  abstract = {Researchers are increasingly applying AI to predict the daily routines of individuals and their corresponding trajectories (referred to as mobility prediction), due to its considerable potential for commercial activities and public administration. However, most previous studies focused exclusively on model theory or design without considering the specific characteristics of human flow trajectories. In this study, we analyzed historical data to identify three factors that play key roles in human mobility. Based on these factors, we developed a temporal-trajectory-based K-nearest neighbor algorithm to predict human flow trajectories. Experimental simulations demonstrated the effectiveness of the proposed scheme when applied to the HuMob Challenge 2024 dataset.},
  booktitle = {Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Human Mobility Prediction Challenge},
  pages = {25–28},
  numpages = {4},
  keywords = {Human mobility, KNN, Machine learning},
  location = {Atlanta, GA, USA},
  series = {HuMob'24}
}
```
## Contact

Feel free to reach out if you have any questions or need assistance:

- 📧 Email: yessir0621@gmail.com
- 🔗 LinkedIn: [An-Syu Li](https://www.linkedin.com/in/an-syu-li-10897a273/)

---
