# HierViT
## Introduction
![HierViT_architecture](https://github.com/user-attachments/assets/5dea178e-be3d-4854-a153-51196ac70626)

Explainability is a highly demanded requirement for applications in high-risk areas such as medicine. Vision Transformers have been mainly limited to attention extraction to provide insight into the model’s reasoning. Our approach combines the high performance of Vision Transformers with the introduction of new explainability capabilities. 

We present HierViT, a Vision Transformer that is inherently interpretable and adapts its reasoning to that of humans. A hierarchical structure is used to processes domain-specific features for prediction. It is interpretable by design, as it reasons the target output with human-defined features that are visualised by exemplary images (prototypes). By using domain knowledge about these decisive features, the reasoning is semantically similar to human reasoning and therefore intuitive. Moreover, attention heatmaps visualise the crucial regions for identifying each feature, thereby providing HierViT with a versatile tool for validating predictions. Evaluated on two medical benchmark datasets, LIDC-IDRI for lung nodule assessment and derm7pt for skin lesion classification, HierViT achieves superior and comparable prediction accuracy, respectively, while providing more explanation that speaks the same language as humans.

This repository includes code for running
- HierViT on LIDC-IDRI, derm7pt, and MNIST, and
- Proto-Caps experiments (https://github.com/XRad-Ulm/Proto-Caps) on LIDC-IDRI and CheXbert.
  
## Installation
1. Clone the repository
          
       git clone https://github.com/your-username/HierViT.git
       cd HierViT
2. Install dependencies
   
   a. Make sure you have Python 3.10 installed.
   
   b. Install PyTorch following the official instructions: https://pytorch.org/get-started/locally/
   
   c. Create and activate the Conda environment:
      
       conda env create -f environment.yml -n HierViT
       conda activate HierViT
  
### Dataset setup
To run the model on the LIDC-IDRI dataset:
1. Download the dataset via the NBIA Data Retriever (select "Images") https://www.cancerimagingarchive.net/collection/lidc-idri/
2. Install pylidc-updated (resolves numpy attribute errors):

       pip3 install pylidc-updated
3. Configure pylidc:
   
   a. Create a configuration file:

       Linux/Mac: /home/[user]/.pylidcrc
       Windows: C:\Users\<YourUser>\pylidc.conf

b) Add the following content, replacing the path with your own dataset location:

    [dicom]
    path = /path/to/big_external_drive/datasets/LIDC-IDRI
    warn = True

4. On first run, the preprocessing pipeline will process and cache the data. This can take 1–2 hours but only needs to be done once per dataset split.
## Running the model
1. Example: Train HierViT on LIDC split 0, using 100% of the attribute annotations:

        --train=True
        --dataset="LIDC"
        --split_number=0
        --base_model="ViT"
        --warmup=2
        --push_step=2
        --resize_shape
        224
        224
        --batch_size=2
        --lam_recon=1
        --lr=0.001
        --shareAttrLabels=1

2. Example: Test trained HierViT model on LIDC-IDRI dataset

        --test=True
        --dataset="LIDC"
        --base_model="ViT"
        --split_number=0
        --model_path="[name of model]_[epoch number].pth"
        --epoch=[epoch number]
        --resize_shape
        224
        224
        --batch_size=2
   
4. Example: Train HierViT on derm7pt dataset:

       --train=True
       --dataset="derm7pt"
       --base_model="ViT"
       --warmup=20
       --push_step=2
       --resize_shape
       224
       224
       --batch_size=16
       --lam_recon=0
       --lr=0.00001

5. Test trained HierViT on derm7pt dataset

       --test=True
       --dataset="derm7pt"
       --base_model="ViT"
       --model_path="[name of model]_[epoch number].pth"
       --epoch=[epoch number]
       --resize_shape
       224
       224
       --batch_size=16
