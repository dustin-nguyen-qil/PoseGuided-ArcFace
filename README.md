# On Face Recognition at Long Distance with Pose-Guided Angular Margin Loss

This repository contains the implementation of the paper "On Face Recognition at Long Distance with Pose-Guided Angular Margin Loss", which proposes a new loss function called Pose-Guided Angular Margin Loss (PGAML) that combines both pose information and angular margin loss to enhance face recognition accuracy at long distances.

## Getting Started
Create a virtual environment with `Python=3.6` using conda:
```python
    $ conda create -n frald python=3.6
    $ conda activate frald
```

Clone the repository:

```python
    $ git clone https://github.com/dustin-nguyen-qil/PoseGuided-ArcFace.git
```

To install the dependencies of the project, run: 

``` python
    $ pip install -r requirements.txt
```
<!-- For the installation of torch using "pip" run the following command

``` python
    $ pip3 install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
``` -->
## Repository content

```
|-- `config`: configuration of the project
|-- `data`: contains files for dataset preparation
|-- `model`: contains source code of model architecture
|-- `output`: contains evaluation results
|-- `trainer`: contains source code for training process
|-- `utils`: contains code to get face pose information
|-- `evaluation.ipynb`: interactive file for evaluation
|-- `evaluate.py`: evaluation file
|-- `train.py`: training file
```
## Training and Implementation details

In this project, we trained two models: `Original ArcFace model` and our proposed `Pose-guided model` on DroneFace dataset with 20 epochs and batch size of 16. 

We use KFold Cross Training with `num_folds=5`, leading to 10 pretrained models in total. Each pretrained model is named as `Model_[type]_Fold[fold_id].pth` where `type = [Original, Pose-guided]`, `fold_id=1,...,4`. 
At each fold, we use 8 IDs for training and the rest 3 IDs for testing. Test images are fed into the trained model after each fold to get the embeddings. The test embeddings would be used for evaluation purpose. 

## Evaluation and Comparision
### Extracted Test embeddings and pre-trained models

You can download the extracted test embeddings and trained models from [here](https://uofh-my.sharepoint.com/:u:/g/personal/dnguy222_cougarnet_uh_edu/ET_B3yewqc9Al2RpjnaSnkMBnkDYDD0EhXEpZg2vhMfP-A?e=C4LMDW). Unzip the file and put `work_space` at the outer most level of the project folder 

```
|-- 
|-- workspace
|  |-- emb: contains extracted test embeddings
|  |-- save: contains trained models
``` 
### Run evaluation
You can run the following command line and refer to `outputs` to see the plots of ROC Curve and CMC Curve results averaged from 5 training folds.

```bash
$ python evaluate.py

```
To see the evaluation results of each training folds, refer to `evaluation.ipynb`.

## Training models

To run training, follow the steps below
### Prepare dataset

Download the DroneFace dataset and its jsons file containing metadata for training and testing from here: [DroneFace dataset](https://uofh-my.sharepoint.com/:u:/g/personal/dnguy222_cougarnet_uh_edu/ERnymCrMfQtFrVoiA4Lwln0BaWR1bo5MERARygtTZnrPzA?e=uWILVu)

Unzip the file, then put `photos_all_faces` inside `data`

```
|-- data
|   |-- photos_all_faces/
|   |-- data_pipe.py
```
### Train DroneFace with Original ArcFace model 

You can run training on the original ArcFace model by going to `config/config.py` and change `conf.pose = False`, then execute

```bash
    $ python train.py -b 16 -e 20 
```
where `-b` is batch size and `-e` is the number of epochs.
### Train DroneFace with Proposed Pose-guided Model

You can run training on the Proposed Pose-guided Model by going to `config/config.py` and change `conf.pose = True`, then execute

```bash
    $ python train.py -b 16 -e 20 
```
where `-b` is batch size and `-e` is the number of epochs.

Trained models of each fold and extracted test embeddings would be automatically saved in `work_space`.
## Acknowledments

This project is based on the following repository and ArcFace paper:
- [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- [Arcface](https://arxiv.org/pdf/1801.07698.pdf)
