import os

def check_missing_files(file_list_path, data_folder):
    # Read the list of files from the train.list
    with open(file_list_path, 'r') as file_list:
        file_names = [line.strip() for line in file_list]

    # Generate the expected file paths based on the given format
    expected_file_paths = [f"{data_folder}/{file_name.split('/')[1]}.npy" for file_name in file_names]

    # Check which files are not present in the folder "03001627_pointClouds"
    missing_files = [file_path for file_path in expected_file_paths if not os.path.exists(file_path)]

    return missing_files

# Example usage
train_list_path = "../data/03001627_train.list"
data_folder = "../data/03001627_pointCloud"

missing_files = check_missing_files(train_list_path, data_folder)

if missing_files:
    print("Missing files:", len(missing_files))
    for missing_file in missing_files:
        print(missing_file)
else:
    print("All files are present.")
