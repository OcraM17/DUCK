# DUCK: Distance-based Unlearning via Centroid Kinematics

![Schema](imgs/Schema.png)
[Marco Cotogni](https://scholar.google.com/citations?user=8PUz5lAAAAAJ&hl=it) ,[Jacopo Bonato](https://scholar.google.com/citations?hl=it&user=tC1GFkUAAAAJ), [Luigi Sabetta](), [Francesco Pelosin](https://scholar.google.com/citations?user=XJ9QvI4AAAAJ&hl=it&authuser=1&oi=ao), [Alessandro Nicolosi]() 

[![arxiv](https://img.shields.io/badge/arXiv-red)]() [![Paper](https://img.shields.io/badge/Journal-brightgreen)]()
## Overview

DUCK is a cutting-edge machine unlearning algorithm designed to enhance privacy in modern artificial intelligence models. Leveraging the power of metric learning, DUCK efficiently removes residual influences of specific data subsets from a neural model's acquired knowledge during training.

## Features

- **Distance-based Unlearning**: DUCK employs distance metrics to guide the removal of samples matching the nearest incorrect centroid in the embedding space.

- **Versatile Performance**: Evaluated across various benchmark datasets, DUCK demonstrates exceptional performance in class removal and homogeneous sampling removal scenarios.

- **Adaptive Unlearning Score (AUS)**: Introducing a novel metric that not only measures the efficacy of unlearning but also quantifies the performance loss relative to the original model.

- **Membership Inference Attack**: DUCK includes a novel membership inference attack to assess its capacity to erase previously acquired knowledge, adaptable to future methodologies.

## Getting Started

### Prerequisites

- [List any prerequisites or dependencies here]

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git

# Navigate to the project directory
cd your-repo

# Install dependencies
[Include any installation commands or requirements]
```

## Code Execution

## Results
![Time](imgs/plot_time.png)
## Citation

```
@misc{2023Duck,
      title={DUCK: Distance-based Unlearning via Centroid Kinematics}, 
      author={Marco Cotogni and Jacopo Bonato and Luigi Sabetta and Francesco Pelosin and Alessandro Nicolosi},
      year={2023},
      eprint={xxx-xxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

