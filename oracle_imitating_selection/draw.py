import os
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

def get_roc_curve(save_figures_path, num_classes, actual_labels, predicted_scores):
    fpr = {}
    tpr = {}
    roc_auc = {}
    color = ["aqua", "darkorange"]

    label_actual = np.array(actual_labels)
    predicted_score = np.array(predicted_scores)

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(label_actual, predicted_score, pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], linestyle="--", color=color[i], label=f"ROC Curve Class {i} (AUC = {roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.title('Oracle Imitating Selection ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(save_figures_path, "roc_curve.png"))
    plt.close()

def get_confusion_matrix(save_figures_path, num_classes, actual_labels, predicted_labels):
    cm = confusion_matrix(actual_labels, predicted_labels, labels=range(num_classes))

    cm_df = pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes))

    seaborn.heatmap(cm_df, annot=True, cmap="Blues", fmt=".1f")

    plt.title("Oracle Imitating Selection ROC curve")
    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")

    plt.savefig(os.path.join(save_figures_path, "confusion_matrix.png"))
    plt.close()

def get_loss_curve(save_figures_path, train_losses, valid_losses):
    plt.plot(train_losses, color="blue", label="train")
    plt.plot(valid_losses, color="red", label="valid")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_figures_path, "lose_curve.png"))
    plt.close()

def get_accuracy_curve(save_figures_path, train_accs, valid_accs):
    plt.plot(train_accs, color="blue", label="train")
    plt.plot(valid_accs, color="red", label="valid")

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_figures_path, "accuracy_curve.png"))
    plt.close()

if __name__ == "__main__":
    pass

