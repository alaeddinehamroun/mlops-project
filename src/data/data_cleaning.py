import os
import hashlib
from PIL import Image

# Remove duplicate images
def compute_hash(file_path):
    """
    Compute the MD5 hash of a file.
    
    Keyword arguments:
    file_path -- file path
    Return: file hash
    """
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.md5(content).hexdigest()

def remove_duplicates(root_dir):
    """
    Remove duplicate images from a folder structure.
    
    Keyword arguments:
    root_dir -- root directory
    Return: -1
    """

    image_hashes = set()

    # Traverse the folder structure
    for dir_path, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Compute the image fingerprint
            file_path = os.path.join(dir_path, filename)
            image_hash = compute_hash(file_path)

            # Check if the image is a duplicate
            if image_hash in image_hashes:
                # Delete the duplicate image
                os.remove(file_path)
                print(f"Removed duplicate: {file_path}")
            else:
                # Add the image hash tothe set
                image_hashes.add(image_hash)
    



# Handle corrupted images
def is_image_corrupted(file_path):
    """
    Check if an image file is corrupted.
    
    Keyword arguments:
    file_path -- file path
    Return: return image status
    """

    try:
        Image.open(file_path).verify()
        return False
    except (IOError, SyntaxError):
        return True

def remove_corrupted_images(root_dir):
    """
    Remove corrupted image files from a folder structure.
        
    Keyword arguments:
    root_dir -- description
    Return: return_description
    """
    # Traverse the folder structure
    for dir_path, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the image file is corrupted 
            file_path = os.path.join(dir_path, filename)
            if is_image_corrupted(file_path):
                # Delete the corrupted image file
                os.remove(file_path)
                print(f"Removed corrupted image: {file_path}")

    
    