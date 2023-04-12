# On Face Recognition at Long Distance with Pose-Guided Angular Margin Loss

This repository contains the implementation of the paper "On Face Recognition at Long Distance with Pose-Guided Angular Margin Loss", which proposes a new loss function called Pose-Guided Angular Margin Loss (PGAML) that combines both pose information and angular margin loss to enhance face recognition accuracy at long distances.

## Getting Started
Create a virtual environment with `Python=3.6` using conda:
```python
    $ conda create -n [name] python=3.7
    $ conda activate [name]
```

Clone the repository:

```python
    $ git clone https://github.com/vuongnd99/PoseGuided-AcrFace.git
```

To install the dependencies of the project, run: 

``` python
    $ pip install -r requirements.txt
```
For the installation of torch using "pip" run the following command

``` python
    $ pip3 install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

## Evaluation and Comparision 

### Pre-trained model

You can download the Original ArcFace and the Proposed Pose-guided models pretrained on DroneFace using the following urls:
- [Original ArcFace](https://uofh-my.sharepoint.com/:u:/g/personal/dnguy222_cougarnet_uh_edu/ES_2h294bKlEljg2k7yyOwABweauZqs1aiWY63ib079jfQ?e=If2GM5)
- [Proposed Pose-guided model](https://uofh-my.sharepoint.com/:u:/g/personal/dnguy222_cougarnet_uh_edu/Efl7CKovvR1HuwVCGdu5OkcB7wjstZRZDqpOvCb6nzF1xw?e=ngBrs0)

Create a folder named `work_space`. Create two subfolders `models` and `save`. Put the pretrained models under `models` 

```
workspace
---> models
    ---> model_final_droneface.pth
    ---> model_final_droneface_pose.pth
---> save
``` 

### Prepare dataset

Download the DroneFace dataset and its jsons file containing metadata for training and testing from here

- [DroneFace with jsons](https://uofh-my.sharepoint.com/:u:/g/personal/dnguy222_cougarnet_uh_edu/EU5O6B4LaqNEoJfAxSlpc64BgAYk1oTPFFCA3o71dQ74OA?e=U337Pb)

Unzip the file, then put `photos_all_faces` inside `data`

```
data
---> photos_all_faces/
    ---> train
    ---> test
    ---> train.json
    ---> test.json
```


### Run evaluation

You can run evaluation to see how our proposed model outperform original ArcFace model by going to `evaluation.ipynb` and run the code cell by cell.
## Training

Trained models will be saved under `work_space/save`, with 
### Train DroneFace with Original ArcFace model 

You can run training on the original ArcFace model by executing

```python
    $ python train.py -b [batch_size] -e [epochs] -d droneface -p False
```
where `-d` specifies DroneFace dataset for training and `-p = False` means we are training with original ArcFace without consideration of Pose.
### Train DroneFace with Proposed Pose-guided Model

You can run training on the proposed Pose-Guided model by executing

```python
    $ python train.py -b [batch_size] -e [epochs] -d droneface -p True
```
where `-d` specifies DroneFace dataset for training and `-p = False`.

## Acknowledments

This project is based on the following repository and ArcFace paper:
- [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- [Arcface](https://arxiv.org/pdf/1801.07698.pdf)
