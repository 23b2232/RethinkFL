# Implementation and comparison of the FPL model for Federated Learning under Domain Shift

This project implements the CVPR 2023 paper 
“Rethinking Federated Learning with Domain Shift: A Prototype View”.         
> Wenke Huang, Mang Ye, Zekun Shi, He Li, Bo Du
> *CVPR, 2023*
> [Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)

The goal is to study how federated models can generalize across heterogeneous domains using prototype-based learning.

## My Contribution

- Implemented Federated Prototypes Learning (FPL) 
- Simulated multi-client federated setup with domain shift
- Conducted Experiments:
1. All 10 participants participating in each training round
2. Random sampling 5-6 [articipants participating in each training round
 to analyze model generalization across domains

## Concept Overview

Federated Learning (FL) typically assumes all clients have data from similar distributions. 
However, in real-world scenarios, data across clients often comes from different domains (domain shift).

This project implements FPL, which improves cross-domain generalization by:
- Learning cluster prototypes representing semantic classes
- Using unbiased prototypes as a common reference across clients
- Applying consistency regularization** to align local models

This helps the global model perform well across heterogeneous data distributions.

## Methodology

1. Each client trains on its local dataset
2. Feature embeddings are extracted
3. Cluster prototypes are computed per class
4. Unbiased prototypes are aggregated globally
5. Consistency regularization aligns local and global representations
6. Model updates are shared across clients

## Key Learnings

- Traditional Federated Learning methods struggle when data distributions differ across clients
- FPL improves performance by:
  - Learning class-level prototypes 
  - Aligning local representations to a shared global structure

This reduces domain-specific bias and improves cross-domain generalization (this can be viewed in the analysis section of the results).

## Citation
```
@inproceedings{HuangFPL_CVPR2023,
    author    = {Huang, Wenke and Mang, Ye and Shi, Zekun and Li, He and Bo, Du},
    title     = {Rethinking Federated Learning with Domain Shift: A Prototype View},
    booktitle = {CVPR},
    year      = {2023}
}
```

