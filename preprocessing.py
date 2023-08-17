import csv
import os

# add 'image_path' to the dataset_information(modified).csv and split into train and test

def main():
    input_file = "dataset_information(modifed).csv"
    output_file_train = "dataset_information(modified)_with_paths_train.csv"
    output_file_test = "dataset_information(modified)_with_paths_valid.csv"

    with open(input_file, 'r') as infile, \
            open(output_file_train, 'w', newline='') as outfile_train, \
            open(output_file_test, 'w', newline='') as outfile_test:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['image_path']

        writer_train = csv.DictWriter(outfile_train, fieldnames=fieldnames)
        writer_train.writeheader()

        writer_test = csv.DictWriter(outfile_test, fieldnames=fieldnames)
        writer_test.writeheader()

        for row in reader:
            slide_name = row['Slide name']
            x_loc = row['x_loc']
            y_loc = row['y_loc']

            patch_names = [
                f"{slide_name}_{x_loc}_{y_loc}.jpg",
                f"{slide_name}_{y_loc}_{x_loc}.jpg"
            ]

            found = False
            
            for patch_name in patch_names:

                # Check if the image is in the "train" or "test" folder
                train_path = f"D:/bryan/oracle_imitating_selection_dataset/train/{slide_name}/{patch_name}"
                test_path = f"D:/bryan/oracle_imitating_selection_dataset/test/{slide_name}/{patch_name}"

                if os.path.exists(train_path):
                    row['image_path'] = train_path
                    writer_train.writerow(row)
                    found = True
                    break
                elif os.path.exists(test_path):
                    row['image_path'] = test_path
                    writer_test.writerow(row)
                    found = True
                    break
            
            if not found:
                print(patch_name)

# if this file is run directly by python
if __name__ == "__main__":
    main()