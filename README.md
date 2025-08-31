# UTL-EADP: An Unsupervised Domain Adversarial Transfer Learning Framework with Extreme Aggregation Discrepancy Prototype-based for Cross-Corpus EEG Emotion Recognition 
*   A Pytorch implementation of our under reviewed paper "UTL-EADP: An Unsupervised Domain Adversarial Transfer Learning Framework with Extreme Aggregation Discrepancy Prototype-based for Cross-Corpus EEG Emotion Recognition".
# Installation
*   Python 3.8
*   Pytorch 2.0.0
*   NVIDIA CUDA 11.8
*   NVIDIA CUDNN 8700
*   Numpy 1.24.3
*   Scikit-learn 0.22.1
*   scipy 1.5.2 
*   GPU NVIDA GeForce RTX 3090
# Databases
*   [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html ""), [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html ""), [SEED-V](https://bcmi.sjtu.edu.cn/~seed/seed-v.html "") 
# Training
*   Data Process Module: DataProcess.py
*   Basic architecture definition file: Basic_Architecture.py
*   Implementation of domain adversarial training: adversarial.py
*   Lmmd module definition file : lmmd.py
*   Cdd module definition file: cdd.py
*   Pipeline of the Lmmd_PL and Cdd_PL: Operation_Lmmd_PL_Cdd_PL.py
*   Mcd_PL model definition file: model_Mcd_PL.py
*   Pipeline of the Mcd_PL : Operation_Mcd_PL.py
*   Trained module in Mcd_PL (source domain: SEED,target domain: SEED_V): model.pth, ada_Classifity.pth and rms_Classifity.pth
# Usage
*   After modify setting (path, etc), just run the main function in the Operation_Lmmd_PL_Cdd_PL.py or Operation_Mcd_PL.py
