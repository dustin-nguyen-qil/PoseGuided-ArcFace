# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import pickle
from data.data_pipe import get_loaders
from config.config import get_config

with open("work_space/emb/test_embeddings_poseguided.pkl",'rb') as f:
    test_embeddings_poseguided = pickle.load(f)

with open("work_space/emb/test_embeddings_original.pkl",'rb') as f:
    test_embeddings_original = pickle.load(f)


"""
Average ROC Curve and AUC Results 
"""

def plot_roc():
    num_folds = 5
    num_classes = 3
    colors = ['red', 'green', 'blue']
    lw = 2

    num_points = 200

    fpr_arr = np.zeros((2, num_classes, num_points))
    tpr_arr = np.zeros((2, num_classes, num_points))
    roc_auc_arr = np.zeros((2, num_classes))

    for fold, (test_embs_original, test_embs_poseguided) in enumerate(zip(
        test_embeddings_original.values(), test_embeddings_poseguided.values()
    )):
        test_labels = np.array([item['label'] for item in test_embs_original])
        test_embs_original_arr = np.array([item['embedding'].squeeze() for item in test_embs_original])
        test_embs_poseguided_arr = np.array([item['embedding'].squeeze() for item in test_embs_poseguided])

        for r, embeddings_set in enumerate([test_embs_original_arr, test_embs_poseguided_arr]):
            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in range(num_classes):
                fpr[f'model_{r}_class_{i}'], tpr[f'model_{r}_class_{i}'], _ = roc_curve(test_labels == i, embeddings_set[:, i])
                roc_auc[f'model_{r}_class_{i}'] = auc(fpr[f'model_{r}_class_{i}'], tpr[f'model_{r}_class_{i}'])
                
            for i in range(num_classes):
                fpr_arr[r, i, :] += np.interp(np.linspace(0, 1, num_points), tpr[f"model_{r}_class_{i}"], fpr[f"model_{r}_class_{i}"])
                tpr_arr[r, i, :] += np.linspace(0, 1, num_points)
                roc_auc_arr[r, i] += roc_auc[f"model_{r}_class_{i}"]

    for r in range(2):
        for i in range(num_classes):
            fpr_arr[r, i, :] /= num_folds
            tpr_arr[r, i, :] /= num_folds
            roc_auc_arr[r, i] /= num_folds

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    models = ['Original ArcFace', 'Proposed Pose-guided model']
    for r, ax in enumerate(axs.flat):
        for i, color in enumerate(colors):
            label = f'Model {r} - Class {i} (AUC = {roc_auc_arr[r, i]:.2f})'
            ax.plot(fpr_arr[r, i, :], tpr_arr[r, i, :], color=color, lw=lw, label=label)

        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(models[r])
        ax.legend(loc="lower right")

    title = f'Average ROC Curve comparison results'
    fig.suptitle(title)
    name = "roc_curve_comparison_results"
    plt.savefig(f"output/{name}.png")


"""
CMC Curve results
"""


"""
Denote the original ArcFace model as Model 1
Denote the Pose-guided model as Model 2
"""

def cmc_curve():
    num_classes = 3  
    num_folds = 5  

    cmc_results = {}  
    for model in ['Model 1', 'Model 2']:
        cmc_results[model] = np.zeros(num_classes)

        for i in range(num_folds):
            embeddings = test_embeddings_original[i] if model == 'Model 1' else test_embeddings_poseguided[i]
            classes = sorted(list(set([emb['label'] for emb in embeddings])))
            num_embeddings = len(embeddings)

            class_embeddings = {c: [] for c in classes}
            for emb in embeddings:
                class_embeddings[emb['label']].append(emb['embedding'])
            class_average_embeddings = {}
            for c, emb_list in class_embeddings.items():
                if len(emb_list) > 0:
                    class_average_embeddings[c] = np.mean(emb_list, axis=0)
                else:
                    class_average_embeddings[c] = None

            similarities = np.zeros((num_embeddings, num_classes))
            for k, emb in enumerate(embeddings):
                for l, c in enumerate(classes):
                    similarities[k, l] = cosine_similarity(emb['embedding'].reshape(1, -1),
                                                        class_average_embeddings[c].reshape(1, -1))

            cmc = np.zeros(num_classes)
            for k in range(num_embeddings):
                sorted_indices = np.argsort(similarities[k, :])[::-1]
                for l in range(num_classes):
                    if classes[sorted_indices[l]] == embeddings[k]['label']:
                        cmc[l:] += 1
                        break

            cmc /= num_embeddings
            cmc_results[model] += cmc

        cmc_results[model] /= num_folds

    # plot average CMC curves
    fig, ax = plt.subplots()
    ax.plot(range(1, num_classes+1), cmc_results['Model 1'], label=f'Original model | R-1 Acc: {cmc_results["Model 1"][0]:.2f}')
    ax.plot(range(1, num_classes+1), cmc_results['Model 2'], label=f'Pose-guided model | R-1 Acc: {cmc_results["Model 2"][0]:.2f}')
    ax.set_xlabel("Rank")
    ax.set_ylabel("CMC Accuracy")
    ax.set_ylim([0, 1])
    ax.legend()
    name = "cmc_curve_comparison_results"
    plt.savefig(f"output/{name}.png")


if __name__ == "__main__":
    plot_roc()
    cmc_curve()