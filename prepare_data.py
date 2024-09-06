import shutil
import pandas as pd
import os
from skimage import io
from skimage.filters import threshold_sauvola, threshold_otsu
import numpy as np
from tqdm import tqdm
import concurrent.futures

from PIL import Image

"""
A class of utility methods to help with data preparation of different datasets. You might have to make some adjustments,
but should give you an idea.

Example usage:
hisir19('/data/hisir19/test_gt.csv', '/data/hisir19/original', '/data/hisir19/binary')
"""

def process_image(row, in_dir, out_dir):
    # Extract filename, writer_id
    
    filename = row['FILENAME'] # in some .csv file there is a typo-> FILENEMA
    writer_id = row['ID']

    # Construct the path for input image and output image
    input_path = os.path.join(in_dir, filename)
    page_id = os.path.splitext(filename)[0]
    output_path = os.path.join(out_dir, f"{writer_id}-IMG_MAX_{page_id}.png")

    try:
        # Load image
        image = io.imread(input_path)

        # Check if the image is already in grayscale
        if len(image.shape) == 3:
            # Convert to grayscale
            image = np.mean(image, axis=2).astype(np.uint8)

        # Apply Sauvola Thresholding
        window_size = 25
        #print("Hi")
        thresh_sauvola = threshold_sauvola(image, window_size=window_size)
        #print("Bye")
        binary_image = image < thresh_sauvola

        # Save the binarized image
    
        io.imsave(output_path, (binary_image * 255).astype(np.uint8))
    except Exception as e:
        print(f"Error processing image {filename}: {e}")


def hisir19(csv_path, in_dir, out_dir):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Creating a list of tasks for each row in the DataFrame
        tasks = [executor.submit(process_image, row, in_dir, out_dir) for _, row in df.iterrows()]
        
        # Progress bar for tasks
        for _ in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
            pass


def process_image_(in_dir, out_dir, filename, thresh="sauvola"):
    # Extract filename, writer_id
    

    # Construct the path for input image and output image
    input_path = os.path.join(in_dir, filename)
    output_path = os.path.join(out_dir, filename)

    try:
        # Load image
        image = io.imread(input_path)

        # Check if the image is already in grayscale
        if len(image.shape) == 3:
            # Convert to grayscale
            image = np.mean(image, axis=2).astype(np.uint8)

        if thresh=="sauvola":
            # Apply Sauvola Thresholding
            window_size = 51
            #print("Hi")
            thresh_sauvola = threshold_sauvola(image, window_size=window_size)
            #print("Bye")
            binary_image = image < (thresh_sauvola) 
        else:
            thresh_otsu = threshold_otsu(image)
            binary_image = image < thresh_otsu
            

        # Save the binarized image
    
        io.imsave(output_path, (binary_image * 255).astype(np.uint8))
    except Exception as e:
        print(f"Error processing image {filename}: {e}")



def cvl(in_dir, train_out_dir, test_out_dir):
    def writername(fname):
        return fname.split("-")[0]
    fnames = os.listdir(in_dir)
    writers = [writername(fname) for fname in fnames]
    writer_counts = {writer: writers.count(writer) for writer in set(writers)}
    
    train_set = [fname for fname in fnames if writer_counts[writername(fname)] == 7]
    test_set = [fname for fname in fnames if writer_counts[writername(fname)] == 5]
    error_set = [fname for fname in fnames if (fname not in train_set and fname not in test_set)]
    
    if len(error_set) > 0:
        print(f"{len(error_set)} files not assigned.", *error_set)
    
    # Ensure output directories exist
    if not os.path.exists(train_out_dir):
        os.makedirs(train_out_dir)
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)

    # Copy files to their respective directories
    for doc in train_set:
        shutil.copy(os.path.join(in_dir, doc), train_out_dir)

    for doc in test_set:
        shutil.copy(os.path.join(in_dir, doc), test_out_dir)
        
        
    
    
def binarize_folder(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Creating a list of tasks for each row in the DataFrame
        tasks = [executor.submit(process_image_, in_dir, out_dir, filename) for filename in os.listdir(in_dir)]
        
        # Progress bar for tasks
        for _ in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
            pass

def process_invert_image(file_path, out_dir):
    try:
        # Load image with PIL to handle different formats, including TIFF
        with Image.open(file_path) as img:
            # Convert to numpy array
            image = np.array(img) * 255

        # Invert image
        image = 255 - image
        image = image.astype(np.uint8)

        # Construct output file path with .png extension
        base_name = os.path.basename(file_path)
        out_file_name = os.path.splitext(base_name)[0] + '.png'
        out_file_path = os.path.join(out_dir, out_file_name)

        Image.fromarray(image).save(out_file_path)
        
    except Exception as e:
        print(f"Error inverting image {file_path}: {e}")
        
        
def invert_all(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Get list of file paths
    file_paths = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]

    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
        tasks = [executor.submit(process_invert_image, file_path, out_dir) for file_path in file_paths]
        
        # Progress bar for tasks
        for _ in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
            pass


from PIL import Image
import os

def process_image(image_path):
    # Load the image
    image = Image.open(image_path)



    grid_size = 112
    
    # Crop a 672x672 window from the image
    o = 200
    global_crop = image.crop((o, o, o+grid_size*6, o+grid_size*6))
    global_crop.save("global.png")

    # Calculate the size of each grid element
    

    # Crop the middle right grid element
    local_crop = global_crop.crop((2 * grid_size, grid_size, 3 * grid_size, 2 * grid_size))
    local_crop.save("local.png")

    # Divide the local image into a 4x4 grid and save each patch
    k = 4
    patch_size = int(grid_size / k)
    if not os.path.exists('patches'):
        os.makedirs('patches')

    for i in range(k):
        for j in range(k):
            patch = local_crop.crop((j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size))
            patch.save(f"patches/{k * i + j}.png")

  
# Example usage:
#hisir19('/data/traven/hisir19/test_gt.csv', '/data/traven/hisir19/wi_comp_19_test', '/data/traven/hisir19/test_binary_26')
#invert_all('/data/traven/hisir19/test_binary', '/data/traven/hisir19/test_binary_c')

#invert_all("/data/traven/2013icdar/test", "/data/traven/2013icdar/test_binary")
#cvl("/data/traven/cvl/binary", "/data/traven/cvl/train_binary", "/data/traven/cvl/test_binary")