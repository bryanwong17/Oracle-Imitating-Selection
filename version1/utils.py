import random
import os
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, precision_score, recall_score

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_class_distribution(data_df, column_name, dataframe_name):
    class_distribution = data_df[column_name].value_counts()
    print(f"Class Distribution for {dataframe_name}:")
    print(class_distribution)

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    train_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    # Set the model to training mode
    model.train()

    for images, targets in train_loader:
       
        if device.type == "cuda":
            images, targets = images.cuda(), targets.cuda()

        # Clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)

        # Calculate the batch loss
        loss = criterion(outputs, targets)
        train_loss += loss.item()

        # Store targets and predictions for computing additional metrics
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy()[:, 1])  # Assuming binary classification

        # Calculate the number of correct predictions and update total count
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update parameters
        optimizer.step()

    # Check if a scheduler exists and use it to update the learning rate
    if scheduler is not None:
        scheduler.step()

    # Calculate acc, auc, precision, and recall using sklearn.metrics
    acc = correct / total
    auc = roc_auc_score(all_targets, all_predictions)
    precision = precision_score(all_targets, np.round(all_predictions))
    recall = recall_score(all_targets, np.round(all_predictions))

    # Calculate and return the average loss and accuracy for the epoch
    return train_loss / len(train_loader), acc, auc, precision, recall

def validate_one_epoch(model, valid_loader, criterion, device):
    valid_loss = 0.0
    total = 0
    correct = 0

    all_targets = []
    all_predictions = []
    
    # set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for images, targets in valid_loader:
            if device.type == "cuda":
                images, targets = images.cuda(), targets.cuda()
            
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)

            # calculate the batch loss
            loss = criterion(outputs, targets)
            valid_loss += loss.item()

            # # calculate the number of correct predictions and update total count
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # store predictions and targets for computing additional metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy()[:, 1])  # Assuming binary classification
        
        # Calculate acc, auc, precision, and recall using sklearn.metrics
        acc = correct / total
        auc = roc_auc_score(all_targets, all_predictions)
        precision = precision_score(all_targets, np.round(all_predictions))
        recall = recall_score(all_targets, np.round(all_predictions))
    
    return valid_loss / len(valid_loader), acc, auc, precision, recall


def test_model(model, test_loader, device):
    predicted_labels = []
    actual_labels = []
    predicted_scores = []

    correct = 0
    total = 0

    with torch.no_grad():

        for images, targets in test_loader:
            if device.type == "cuda":
                images, targets = images.cuda(), targets.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())

            actual_labels.extend(targets.cpu().numpy())

            # get the predicted score based on the predicted label
            output_sigmoid = torch.softmax(output, dim=1) # batch_size x num_classes
            scores = output_sigmoid[range(output_sigmoid.shape[0]), predicted] # batch_size x 1
            predicted_scores.extend(scores.cpu().numpy())

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return actual_labels, predicted_labels, predicted_scores

def fit_gpu(model, save_results_path, epochs, device, criterion, optimizer, scheduler, train_loader, valid_loader=None):

    # keeping track of losses as it happen
    train_losses = []
    train_accs = []
    train_aucs = []
    train_precisions = []
    train_recalls = []

    valid_losses = []
    valid_accs = []
    valid_aucs = []
    valid_precisions = []
    valid_recalls = []

    for epoch in range(1, epochs + 1):

        # train_loader
        print(f"{'='*50}")
        print(f"EPOCH {epoch} - TRAINING...")
        train_loss, train_acc, train_auc, train_precision, train_recall = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        with open(os.path.join(save_results_path, f"full_log.txt"), "a") as f:
            f.write(f"[TRAIN] EPOCH {epoch} - LOSS: {train_loss:.4f}, ACC: {train_acc:.3f}, AUC: {train_auc:.3f}, Precision: {train_precision:.3f}, Recall: {train_recall:.3f}\n")
            print(f"[TRAIN] EPOCH {epoch} - LOSS: {train_loss:.4f}, ACC: {train_acc:.3f}, AUC: {train_auc:.3f}, Precision: {train_precision:.3f}, Recall: {train_recall:.3f}\n")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_aucs.append(train_auc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)

        # valid_loader
        if valid_loader is not None:
            print(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_acc, valid_auc, valid_precision, valid_recall = validate_one_epoch(
                model, valid_loader, criterion, device
            )

            with open(os.path.join(save_results_path, f"full_log.txt"), "a") as f:
                f.write(f"[VALIDATION] EPOCH {epoch} - LOSS: {valid_loss:.4f}, ACC: {valid_acc:.3f}, AUC: {valid_auc:.3f}, Precision: {valid_precision:.3f}, Recall: {valid_recall:.3f}\n\n")
                print(f"[VALIDATION] EPOCH {epoch} - LOSS: {valid_loss:.4f}, ACC: {valid_acc:.3f}, AUC: {valid_auc:.3f}, Precision: {valid_precision:.3f}, Recall: {valid_recall:.3f}\n\n")

            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            valid_aucs.append(valid_auc)
            valid_precisions.append(valid_precision)
            valid_recalls.append(valid_recall)

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "valid_auc": valid_auc,
            "valid_precision": valid_precision,
            "valid_recall": valid_recall,
            "criterion": criterion,
            "optimizer": optimizer
        }

        # Convert the log dictionary to a formatted string
        log_str = (
            f"Epoch: {log['epoch']}\n"
            f"Train Loss: {log['train_loss']:.4f}\n"
            f"Train ACC: {log['train_acc']:.3f}\n"
            f"Train AUC: {log['train_auc']:.3f}\n"
            f"Train Precision: {log['train_precision']:.3f}\n"
            f"Train Recall: {log['train_recall']:.3f}\n"
            f"Validation Loss: {log['valid_loss']:.4f}\n"
            f"Validation ACC: {log['valid_acc']:.3f}\n"
            f"Validation AUC: {log['valid_auc']:.3f}\n"
            f"Validation Precision: {log['valid_precision']:.3f}\n"
            f"Validation Recall: {log['valid_recall']:.3f}\n"
            f"Criterion: {log['criterion']}\n"
            f"Optimizer: {log['optimizer']}"
        )

        # write the log string to a .txt file
        save_log_path = os.path.join(save_results_path, "logs")
        if not os.path.exists(save_log_path):
            os.makedirs(save_log_path)
        with open(os.path.join(save_log_path, f"log_{log['epoch']}.txt"), "w") as f:
            f.write(log_str)

        # save the model weights for every epoch
        save_weights_path = os.path.join(save_results_path, "weights")
        if not os.path.exists(save_weights_path):
            os.makedirs(save_weights_path)
        torch.save(model.state_dict(), os.path.join(save_weights_path, f"weights_{epoch}.pth"))

    # load the last epoch model
    last_epoch = epochs
    model_state_dict_path = os.path.join(save_weights_path, f"weights_{last_epoch}.pth")
    model.load_state_dict(torch.load(model_state_dict_path))

    return model, train_losses, valid_losses, train_accs, valid_accs