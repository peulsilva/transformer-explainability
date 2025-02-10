
# **Transformer Explainability - MAP583**  

This repository contains the code used for the experiments and analysis presented in the talk *"Transformer Explainability: A Survey"* by Pedro Silva and Sabrina Sartori. The project explores different interpretability techniques for Transformer-based models, particularly in NLP and vision tasks.  

---

## **Table of Contents**  
- [**Transformer Explainability - MAP583**](#transformer-explainability---map583)
  - [**Table of Contents**](#table-of-contents)
  - [**Introduction**](#introduction)
  - [**Setup \& Installation**](#setup--installation)
    - [**Requirements**](#requirements)
    - [**Installation**](#installation)
  - [**Approaches**](#approaches)
    - [**1. Attention-Based Metrics**](#1-attention-based-metrics)
    - [**2. Gradient-Based Metrics**](#2-gradient-based-metrics)
  - [**Experiments**](#experiments)
    - [**1. Qualitative Analysis**](#1-qualitative-analysis)
    - [**2. Perturbation Metrics**](#2-perturbation-metrics)
    - [**3. Probing Attention Heads**](#3-probing-attention-heads)
  - [**Results**](#results)
    - [**AG News Task**](#ag-news-task)
    - [**ImageNet Task**](#imagenet-task)
  - [**Conclusion**](#conclusion)
  - [**Authors**](#authors)

---

## **Introduction**  
Transformer models have achieved state-of-the-art performance across various NLP and vision tasks. However, their interpretability remains a challenge due to their deep architectures and large parameter counts. This project explores different techniques to analyze model behavior, detect biases, and ensure fairness using:  
- **Attention-based explainability metrics**  
- **Gradient-based attribution methods**  
- **Perturbation-based impact analysis**  

---

## **Setup & Installation**  
### **Requirements**  
- Python 3.8+  
- PyTorch  
- NumPy  
- Matplotlib  
- Transformers (Hugging Face)  

### **Installation**  
Clone the repository and install the dependencies:  
```bash
git clone https://github.com/peulsilva/transformer-explainability
cd transformer-explainability
pip install -r requirements.txt
```


---

## **Approaches**  

### **1. Attention-Based Metrics**  
- **Raw Attention**: Uses raw attention weights from a layer.  
- **Attention Rollout**: Incorporates residual connections for a better explanation.  
- **Influence**: Propagates attention weights using vector norms.  

### **2. Gradient-Based Metrics**  
- **Gradient Rollout**: Computes the gradient of an input token/image w.r.t. a target class to highlight influential regions.  

---

## **Experiments**  

### **1. Qualitative Analysis**  
- Evaluates explainability metrics from a human perspective.  
- Compares **Attention Rollout** vs. **Influence** vs. **Gradient Rollout** on different datasets.  

### **2. Perturbation Metrics**  
- Tests the sensitivity of models by removing key tokens and evaluating the impact.  
- Tasks:  
  - **AG News Classification**  
  - **ImageNet Classification**  

### **3. Probing Attention Heads**  
- Identifies the most relevant attention heads using a classifier trained on extracted attention scores.  

---

## **Results**  

### **AG News Task**  
| Metric            | Positive perturbation AUC (Lower = Better) | Negative perturbation AUC (Higher = Better) |
| ----------------- | ------------------------------------------ | ------------------------------------------- |
| Attention Rollout | 0.60                                       | 0.79                                        |
| Influence         | 0.57                                       | 0.80                                        |
| Gradient Rollout  | 0.56                                       | 0.81                                        |

### **ImageNet Task**  
| Metric            | Positive perturbation AUC (Lower = Better) | Negative perturbation AUC (Higher = Better) |
| ----------------- | ------------------------------------------ | ------------------------------------------- |
| Attention Rollout | 0.43                                       | 0.70                                        |
| Influence         | 0.40                                       | 0.72                                        |
| Gradient Rollout  | 0.36                                       | 0.73                                        |

Gradient-based methods showed superior performance in identifying important tokens. Influence remains a computationally efficient alternative for large-scale models.  

---

## **Conclusion**  
- **Gradient Rollout** provides the most accurate interpretability insights.  
- **Influence** is computationally lighter while still outperforming Attention Rollout.  
- **Probing Attention Heads** helps identify the most critical layers for decision-making.  

This project serves as a foundation for further research in Transformer explainability. Contributions and discussions are welcome!  

---

## **Authors**  
- **Pedro Silva**  
- **Sabrina Sartori**  

