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

## Project Setup

### Pre-trained model

The pretrained model of original ArcFace has been provided in the <b> work_space/model </b> and <b> work_space/save </b> folder. If you want to download the models you can follow the following url:
- [IR-SE50 @ Onedrive](https://onedrive.live.com/?authkey=%21AOw5TZL8cWlj10I&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21835&parId=root&action=locate)
- [Mobilefacenet @ OneDrive](https://onedrive.live.com/?authkey=%21AIweh1IfiuF9vm4&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21836&parId=root&o=OneUp)


## Training

### Prepare dataset
If you want to run training, download the refined dataset: (emore recommended)

- [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ)
- [emore dataset @ Dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)


After unzip the files to 'data' path, run :

  ```python
    $ python data/prepare_data.py
  ```

This will take few hours depending on your system configuration.

After the execution, you should find following structure:

```
faces_emore/
            ---> agedb_30
            ---> calfw
            ---> cfp_ff
            --->  cfp_fp
            ---> cfp_fp
            ---> cplfw
            --->imgs
            ---> lfw
            ---> vgg2_fp
```

### Training
Execute the following command for training
```python
    $ python train.py -n [network_mode] -b [batch_size] -e [epochs]
```
where `network_mode = [ir_se, mobilefacenet]`. For other training options, run

```python
    $ python train.py --help
```
## Acknowledments

This project is based on the following repository and ArcFace paper:
- [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- [Arcface](https://arxiv.org/pdf/1801.07698.pdf)
