        
        markdown
        
    
  
      # Neural Network Stability Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📌 Overview
This repository contains a quantitative framework for analyzing the stability of Deep Neural Networks (DNNs) when subjected to structural weight perturbations. The code utilizes a standard Multi-Layer Perceptron (MLP) trained on the MNIST dataset to model the relationship between synaptic weight integrity and functional inference accuracy.

Key analyses included:
1.  **Sensitivity Analysis:** Measuring the non-linear degradation of accuracy in response to Gaussian noise injection.
2.  **Recovery Dynamics:** Quantifying the computational cost (epochs/gradient steps) required to restore baseline performance after a structural perturbation event.

## 🚀 Usage

### Prerequisites
*   Python 3.8+
*   PyTorch
*   Torchvision
*   Matplotlib
*   Numpy

### Installation
```bash
git clone https://github.com/hossein-noorollahi/Neural-Stability-Analysis.git
cd Neural-Stability-Analysis
pip install -r requirements.txt
    
    
  
  
Running the Analysis

        
        bash
        
    
  
      python Simulation.py
    
    
  
  
The script will automatically download the MNIST dataset (if not present locally), train the model, perform the perturbation analysis, and generate the following plots:


Sensitivity_Analysis.png: Accuracy vs. Noise Standard Deviation.

Recovery_Cost.png: Retraining trajectory comparing perturbed vs. baseline models.


📄 License

This project is open-source and available under the MIT License.

    
    
  
  

