import os.path as osp
from config.config import get_config
from trainer.Learner import face_learner
import matplotlib.pyplot as plt
import argparse, pickle
from data.data_pipe import get_loaders

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode",default='droneface', type=str)
    parser.add_argument("-fold", "--num_folds", help="number of folds in kfold", default=4, type=int)

    args = parser.parse_args()

    conf = get_config()
    
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth   

    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode

    num_folds = args.num_folds

    loaders, class_num = get_loaders(conf, num_folds)

    epoch_loss_per_fold_original = {}
    epoch_loss_per_fold_poseguided = {}
    test_embeddings_original = {}
    test_embeddings_poseguided = {}

    if not conf.pose:
        print("===== Training original ArcFace model on DroneFace =====")

        for fold, (train_loader, test_loader) in enumerate(loaders):
            print()
            print(f"Fold: {fold} started ============")
            epoch_loss_per_fold_original[fold] = []
            learner = face_learner(conf, class_num=class_num)
            epoch_loss_per_fold_original, test_embeddings = learner.train(
                conf, fold, args.epochs, train_loader, test_loader, epoch_loss_per_fold_original
            )
            test_embeddings_original[fold] = test_embeddings

        with open(osp.join(conf.work_path, "emb/test_embeddings_original.pkl"), 'wb') as f:
            pickle.dump(test_embeddings_original, f)

    else:
        print("===== Training proposed Pose-guided model on DroneFace =====")

        for fold, (train_loader, test_loader) in enumerate(loaders):
            print()
            print(f"Fold: {fold} started ============")
            epoch_loss_per_fold_poseguided[fold] = []
            learner = face_learner(conf, class_num=class_num)
            epoch_loss_per_fold_poseguided, test_embeddings = learner.train(
                conf, fold, args.epochs, train_loader, test_loader, epoch_loss_per_fold_poseguided
            )
            test_embeddings_poseguided[fold] = test_embeddings

        with open(osp.join(conf.work_path, "emb/test_embeddings_poseguided.pkl"), 'wb') as f:
            pickle.dump(test_embeddings_poseguided, f)
    

# # plot loss values for each fold 
# for fold, loss_values in epoch_loss_per_fold.items():
#     plt.plot(range(1, len(loss_values)+1), loss_values, label=f'Fold {fold}')

# plt.title('Training Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('output/training_loss.png')
