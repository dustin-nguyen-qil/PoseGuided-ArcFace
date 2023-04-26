from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.data_path = Path('data')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'droneface'
    conf.pose = True # train model on DroneFace with Pose
    conf.droneface_folder = conf.data_path/'photos_all_faces'/'all_data'
    conf.droneface_train_json = conf.data_path/'photos_all_faces'/'all_data.json'

    conf.batch_size = 32 # retrain with droneface 
#   conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [12,15,18]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()    
        conf.num_folds = 4
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf