import concurrent.futures
from tqdm import tqdm
import numpy as np

def read_obj_file(file_path):
    vertices = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = [float(coord) for coord in line.split()[1:]]
                vertices.append(vertex)

    return np.array(vertices)

def process_file(file_name, data_folder, target_size=319348):
    obj_file_path = f"{data_folder}/{file_name}/models/model_normalized.obj"
    point_cloud = read_obj_file(obj_file_path)

    # If the point cloud size is less than the target size, pad with zeros
    if point_cloud.shape[0] < target_size:
        padding_size = target_size - point_cloud.shape[0]
        padding = np.zeros((padding_size, point_cloud.shape[1]))
        point_cloud = np.vstack([point_cloud, padding])

    # Process the point cloud as needed
    np.save(f"{data_folder}/03001627_pointCloud/{file_name.split('/')[1]}.npy", point_cloud[:target_size])

    return point_cloud.shape

def process_files_parallel(file_list_path, data_folder):
    max_size = 0  # Variable to store the maximum size

    with open(file_list_path, 'r') as file_list:
        file_names = [line.strip() for line in file_list]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_name, data_folder) for file_name in file_names]

        # Use tqdm to create a progress bar
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    max_size = max(max_size, result[0])  # Assuming result is a tuple containing the shape
                    # print(f"Processed file: {result}")
                except Exception as e:
                    print(f"An error occurred: {e}")
                finally:
                    pbar.update(1)

    return max_size

# Example usage
train_list_path = "data/03001627_train.list"
data_folder = "data"
max_point_cloud_size = process_files_parallel(train_list_path, data_folder)
print("max point cloud size: ", max_point_cloud_size)

# 319348
