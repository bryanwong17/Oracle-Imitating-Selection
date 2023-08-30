import numpy as np
import pandas as pd
import argparse
import os
import sys

import torch
import torch.nn as nn

draw_path = os.path.join(os.path.dirname(__file__), "..")  # Adjust the number of ".." as needed
sys.path.append(draw_path)

from datetime import datetime
from torchvision.models import resnet18, vgg16, densenet201
from sklearn.metrics import accuracy_score

from utils_concat_meta_info_combined_model import seed_everything, print_class_distribution, test_model, fit_gpu
from draw import get_loss_curve, get_accuracy_curve, get_confusion_matrix, get_roc_curve
from dataset_concat_meta_info_combined_model import PatchesDataset

# To run with the default seed (0):
# python main.py

# To run with multiple seeds:
# python main.py --seeds 0 10 100 1000 10000

# To run with the default seed (0), using a weighted random sampler, and downsampled validation (balanced):
# python main.py --weighted_random_sampler --downsampled_valid

# To run with the default seed (0), using a weighted random sampler, downsampled validation (balanced), and a specific model (e.g., vgg16):
# python main.py --upsampled_train --downsampled_valid --model=vgg16

# To run with the default seed (0), using a weighted random sampler, downsampled validation (balanced), a specific model (e.g., vgg16), and a specific scheduler (e.g., step_lr):
# python main.py --upsampled_train --downsampled_valid --model=vgg16 --use_sched --sched=step_lr

parser = argparse.ArgumentParser(description="Oracle Imitating Selection Model Concat with Meta Info Experiment")
parser.add_argument("--seeds", nargs='+', type=int, default=[0],
                    help="seed numbers for reproducibility")
parser.add_argument("--train_csv", default="dataset_information(modified)_with_paths_train.csv", type=str,
                    help="location of train csv file")
parser.add_argument("--valid_csv", default="dataset_information(modified)_with_paths_valid.csv", type=str,
                    help="location of test csv file")
parser.add_argument("--save_results_path", default="results", type=str,
                    help="save folder path")
parser.add_argument("--weighted_random_sampler", action="store_true",
                    help="Whether to define the weights for each class which would be inversely proportional to the number of samples for each class")
parser.add_argument("--upsampled_train", action="store_true",
                    help="over sampling class 1 (oracle selection) to match the number of class 0 (non oracle selection) in training dataset")
parser.add_argument("--downsampled_valid", action="store_true",
                    help="under sampling class 0 (non oracle selection) to match the number of class 1 (oracle selection) in validation dataset")
parser.add_argument("--image_size", default=256, type=int,
                    help="size of images (patches)")
parser.add_argument("--model", default="resnet18", type=str,
                    choices=["resnet18", "vgg16", "densenet201"],
                    help="name of model")
parser.add_argument("--num_classes", default=2, type=int,
                    help="whether oracle selection or not")
parser.add_argument("--epochs", default=100, type=int,
                    help="number of epochs during training")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size to use during training")
parser.add_argument("--optim", default="sgd", type=str,
                    choices=["sgd", "adam"],
                    help="which optimizer to use during training")
parser.add_argument('--momentum', default=0.9, type=float,
                    help="momentum of the optimizer")
parser.add_argument("--use_sched", action="store_true",
                    help="use scheduler during training if needed")
parser.add_argument("--sched", default="exponential_lr", type=str,
                    choices=["step_lr", "exponential_lr"],
                    help="which scheduler to use during training")
parser.add_argument("--lr", default=0.01, type=float,
                    help="learning rate")
parser.add_argument('--step_size', default=10, type=int,
                    help="each step size updates the learning rate")
parser.add_argument('--gamma', default=0.8, type=float,
                    help="multiply the learning rate by gamma after step size")

args = parser.parse_args()

class OracleImitationModel(torch.nn.Module):
    def __init__(self, feature_extractor, input_dim, output_dim):
        super(OracleImitationModel, self).__init__()
        self.feature_extractor = feature_extractor
    
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 64)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(64, output_dim)
    
    def forward(self, x, meta_info):
        x = self.feature_extractor(x).squeeze() # torch.Size([BS, 512])

        # Separate the one-hot encoded tensors and scores tensor
        one_hot_tensors = meta_info[:-1] # [torch.Size([BS, 3]), torch.Size([BS, 3])]
        patch_conf_score_tensors = meta_info[-1] # torch.Size([BS])

        concatenated_one_hot = torch.cat(one_hot_tensors, dim=1) # torch.Size([BS, 6])

        x = torch.cat((x, concatenated_one_hot, patch_conf_score_tensors.unsqueeze(1)), dim=1) # torch.Size([BS, 519])

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def main():

    for seed in args.seeds:
        print(f"SEED: {seed}, Model: {args.model}")

        # for reproductibility
        seed_everything(seed)

        # extend save folder path
        args.save_results_path = os.path.join("../results", args.save_results_path, f"seed_{seed}", f"{args.model}")
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        train_df = pd.read_csv(os.path.join("../dataset", args.train_csv))

        if args.weighted_random_sampler:

            class_sample_counts = np.array(
                [len(train_df[train_df["oracle_selection"] == y]) for y in np.unique(train_df["oracle_selection"])]
            )
            num_samples = sum(class_sample_counts)

            class_weight = [num_samples / class_sample_counts[i] for i in range(len(class_sample_counts))]
         
            samples_weight = np.array([class_weight[y] for y in train_df["oracle_selection"]])
            samples_weight = torch.from_numpy(samples_weight)

            # replacement=False -> we will only see the sample once when iterates through the entire dataset
            sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight.type("torch.DoubleTensor"), num_samples=len(samples_weight), replacement=True)

            train_dataset = PatchesDataset(train_df, args.image_size, is_train=True)
    
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                # shuffle=True,
                sampler=sampler
            )

            # # define criterion based on the class weight
            # criterion = nn.CrossEntropyLoss(weight=class_weight)

        ## train dataset: upsampling class 1 (oracle selection) to balance the distribution with class 0 (non oracle selection) ##
        if args.upsampled_train:

            class_0_samples = train_df[train_df["oracle_selection"] == 0]
            class_1_samples = train_df[train_df["oracle_selection"] == 1]

            # determine the number of samples to generate for class 1 (same as class 0)
            num_samples_to_generate = len(class_0_samples) - len(class_1_samples)

            # repeat the class 1 samples to match the size of class 0
            upsampled_class_1_samples = class_1_samples.sample(n=num_samples_to_generate, replace=True, random_state=seed)

            # combine the upsampled class 1 samples with class 0 samples
            train_df = pd.concat([class_0_samples, class_1_samples, upsampled_class_1_samples])

            # Shuffle the combined DataFrame
            train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

            # Save balanced_valid_df to a CSV file
            train_df.to_csv(os.path.join(args.save_results_path, "upsampled_train.csv"), index=False)

            train_dataset = PatchesDataset(train_df, args.image_size, is_train=True)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                shuffle=True
            )

        print_class_distribution(train_df, "oracle_selection", "Training Dataset")

        valid_df = pd.read_csv(os.path.join("../dataset", args.valid_csv))

        # define architecture
        if args.model == "resnet18":
            model = resnet18(pretrained=True)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        else:
            # it will be added soon
            pass
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = OracleImitationModel(feature_extractor, input_dim=519, output_dim=2)

        # define criterion
        if args.weighted_random_sampler:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float).to(device))
        else:
            criterion = nn.CrossEntropyLoss()
    
        model.to(device)

        # select optimizer
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
        elif args.optim == "adam":
            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=0.001)
        
        # define scheduler
        if args.use_sched:
            if args.sched == "exponential_lr":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
            elif args.sched == "step_lr":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        else:
            scheduler = None
            
        start_time = datetime.now()
        print(f"Start Time: {start_time}")

        last_epoch_model, train_losses, valid_losses, train_accs, valid_accs = fit_gpu(
            image_size=args.image_size,
            batch_size=args.batch_size,
            model=model,
            save_results_path=args.save_results_path,
            epochs=args.epochs,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            valid_df=valid_df
        )

        print(f"Training Execution time: {datetime.now() - start_time}")
        print("Saving Model")
        save_weights_path = os.path.join(args.save_results_path, "weights")
        torch.save(last_epoch_model.state_dict(), os.path.join(save_weights_path, "weights_last_epoch.pth"))

        save_figures_path = os.path.join(args.save_results_path, "figures")
        if not os.path.exists(save_figures_path):
            os.makedirs(save_figures_path)

        # plot train and valid loss
        get_loss_curve(save_figures_path, train_losses, valid_losses)

        # plot train and valid acc
        get_accuracy_curve(save_figures_path, train_accs, valid_accs)

        # # TESTING
        # save_best_weights_path = os.path.join(save_weights_path,  "weights_last_epoch.pth")
        # model.load_state_dict(torch.load(save_best_weights_path))
        # actual_labels, predicted_labels, predicted_scores = test_model(model, valid_loader, device)
        # print(f"Testing Execution time: {datetime.now() - start_time}")

        # # accuracy score
        # print("Accuracy Score:")
        # print(accuracy_score(actual_labels, predicted_labels) * 100)

        # save_figures_path = os.path.join(args.save_results_path, "figures")
        # if not os.path.exists(save_figures_path):
        #     os.makedirs(save_figures_path)

        # # roc curve
        # get_roc_curve(save_figures_path, args.num_classes, actual_labels, predicted_scores)
        # # confusion matrix
        # get_confusion_matrix(save_figures_path, args.num_classes, actual_labels, predicted_labels)

# if this file is run directly by python
if __name__ == "__main__":
    main()
