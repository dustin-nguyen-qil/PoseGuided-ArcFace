from model.model import Backbone, Arcface, MobileFaceNet, l2_norm, PoseArcFace
import os.path as osp
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils.utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math

#--------------------Training Config -------------- 
class face_learner(object):
    def __init__(self, conf, class_num, inference=False):

        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)

            if not conf.train_from_scratch:
                self.load_state(conf, 'final.pth', from_save_folder=True, model_only=True)

            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        
        if not inference:
            self.milestones = conf.milestones
            self.class_num = class_num 

            self.step = 0
            if conf.pose:
                self.head = PoseArcFace(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            else:
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)

            print('Model head generated')

            

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')    
            print()
            
        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
            ('model:{}_step:{}_{}.pth'.format(get_time(), self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_step:{}_{}.pth'.format(get_time(), self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer:{}_step:{}_{}.pth'.format(get_time(), self.step, extra)))
    
    
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path            
        self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
        print("load model successfully")
        
        
    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)          
          
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses    

    def train(self, conf, fold, epochs, train_loader, test_loader, epoch_loss_per_fold):
        self.model.train()
        
        """
        Training
        """
        for e in range(epochs):
            running_loss = 0.
            num_corrects = 0
            total_samples = 0
            print(f'Epoch {e} in fold {fold} started')
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()                                 
            for imgs, labels, yaws, heights in tqdm(iter(train_loader)):
                if fold == 0:
                    labels -= 2
                elif fold == 1:
                    mask = labels > 3
                    labels[mask] -= 2
                elif fold == 2:
                    mask = labels > 5
                    labels[mask] -= 2
                elif fold == 3:
                    mask = labels > 7
                    labels[mask] -= 2
        
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                if conf.pose:
                    thetas = self.head(embeddings, labels, yaws, heights)
                else:
                    thetas = self.head(embeddings, labels)

                _, predicted = torch.max(thetas.data, 1)
                num_corrects += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

            epoch_loss = running_loss / len(train_loader)
            epoch_loss_per_fold[fold].append(epoch_loss)

            train_accuracy = num_corrects / total_samples

            if e == epochs - 1:
                print(f"==== Done training for fold {fold} | Training Accuracy: {train_accuracy*100:.3f}% =====")
            else:
                print(f"Training loss: {epoch_loss:.3f} | Training Accuracy: {train_accuracy*100:.3f}%")
        print()
        """
        Validation
        """
        print(f"==== Start Inference for fold {fold} =====")
        self.model.eval()
        # running_test_loss = 0.
        # num_test_corrects, total_test_samples = 0, 0 

        test_embeddings = []
        for img, label, _, _ in tqdm(iter(test_loader)):
            if label == 10:
                label -= 8
            else:
                label = label - 2*fold
            img = img.to(conf.device)

            with torch.inference_mode():
                embedding = self.model(img)
            test_embeddings.append({'embedding': embedding.detach().cpu().numpy(), 'label': label.numpy()[0]})

        
        print(f"====== End fold {fold} =======")
        print()
        if conf.pose:       
            torch.save(self.model.state_dict(), osp.join(conf.save_path, f'Model_PoseGuided_Fold{fold}'))
        else:
            torch.save(self.model.state_dict(), osp.join(conf.save_path, f'Model_Original_Fold{fold}'))

        return epoch_loss_per_fold, test_embeddings

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                
            params['lr'] /= 10
        print(self.optimizer)
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               