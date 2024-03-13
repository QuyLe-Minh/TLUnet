# TLUnet: Efficient and Accurate 3D Medical Image Segmentation

## Architecture overview of UNETR++
![Architecture overview](https://github.com/QuyLe-Minh/TLUnet/assets/92782164/0412fc09-3052-4d31-b6f2-0bded3e50645)


<hr />


## Installation
The code is tested with PyTorch 1.11.0 and CUDA 11.3. After cloning the repository, follow the below steps for installation,

1. Create and activate conda environment
```shell
conda create --name unetr_pp python=3.8
conda activate unetr_pp
```
2. Install PyTorch and torchvision
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install other dependencies
```shell
pip install -r requirements.txt
```
<hr />


## Dataset
We follow the same dataset preprocessing as in [UNETR++](https://github.com/Amshaker/unetr_plus_plus). We conducted extensive experiments on Synapse

The dataset folders for Synapse should be organized as follows: 

```
./DATASET_Synapse/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task002_Synapse
       ├── unetr_pp_cropped_data/
           ├── Task002_Synapse
 ```
 
 The dataset folders for ACDC should be organized as follows: 

```
./DATASET_Acdc/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task001_ACDC
       ├── unetr_pp_cropped_data/
           ├── Task001_ACDC
 ```

 
Please refer to [Setting up the datasets](https://github.com/282857341/nnFormer) on nnFormer repository for more details.
Alternatively, you can download the preprocessed dataset for [Synapse](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/EbHDhSjkQW5Ak9SMPnGCyb8BOID98wdg3uUvQ0eNvTZ8RA?e=YVhfdg)

## Training
The following scripts can be used for training our UNETR++ model on the datasets:
```shell
bash training_scripts/run_training_synapse.sh
```

<hr />

## Evaluation

To reproduce the results of TLUnet: 

1- Download [Synapse weights](https://drive.google.com/file/d/13JuLMeDQRR_a3c3tr2V2oav6I29fJoBa) and paste ```model_final_checkpoint.model``` in the following path:
```shell
tlunet/evaluation/unetr_pp_synapse_checkpoint/unetr_pp/3d_fullres/Task002_Synapse/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/
```
Then, run 
```shell
bash evaluation_scripts/run_evaluation_synapse.sh
```


<hr />

## Acknowledgement
This repository is built based on [UNETR++](https://github.com/Amshaker/unetr_plus_plus) repository.

