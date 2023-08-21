import os
import re
import matplotlib.pyplot as plt

def get_loss_curve(save_figures_path, train_losses, valid_losses):
    plt.plot(train_losses, color="blue", label="train")
    plt.plot(valid_losses, color="red", label="valid")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_figures_path, "loss_curve.png"))
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

def main():

    ## LOSS CURVE ##

    # # Read the contents of the log file
    # log_filename = "results/results_train_weighted_random_sampler_concat_meta_info/seed_0/resnet18/full_log.txt"
    # with open(log_filename, "r") as log_file:
    #     log_contents = log_file.read()

    # # Extract train and validation losses using regular expressions
    # train_losses = re.findall(r"\[TRAIN\] .* - ACC: ([\d.]+),", log_contents)
    # valid_losses = re.findall(r"\[VALIDATION\] .* - ACC: ([\d.]+),", log_contents)

    # # Convert extracted losses to floats
    # train_losses = [float(loss) for loss in train_losses]
    # valid_losses = [float(loss) for loss in valid_losses]

    # # Provide the path where you want to save the figure
    # save_figures_path = "results/results_train_weighted_random_sampler_concat_meta_info/seed_0/resnet18/figures"

    # # Call the function to generate the loss curve plot
    # get_loss_curve(save_figures_path, train_losses, valid_losses)

    ## ACCURACY CURVE ##

    # Read the contents of the log file
    log_filename = "results/results_train_weighted_random_sampler_concat_meta_info/seed_0/resnet18/full_log.txt"
    with open(log_filename, "r") as log_file:
        log_contents = log_file.read()

    # Extract train and validation losses using regular expressions
    train_accs = re.findall(r"\[TRAIN\] .*ACC: ([\d.]+)", log_contents)
    valid_accs = re.findall(r"\[VALIDATION\] .*ACC: ([\d.]+)", log_contents)

    # Convert extracted losses to floats
    train_accs = [float(acc) for acc in train_accs]
    valid_accs = [float(acc) for acc in valid_accs]

    # Provide the path where you want to save the figure
    save_figures_path = "results/results_train_weighted_random_sampler_concat_meta_info/seed_0/resnet18/figures"

    # Call the function to generate the loss curve plot
    get_accuracy_curve(save_figures_path, train_accs, valid_accs)

if __name__ == "__main__":
    main()
