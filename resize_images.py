from PIL import Image
import os

def resize_images_to_224(source_dir, target_dir):
    """
    Resize all images in the source directory to 224x224 pixels and save them to the target directory.

    Args:
        source_dir (str): Path to the source directory containing images.
        target_dir (str): Path to the target directory to save resized images.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_dir):
        # Full path to the source image
        source_path = os.path.join(source_dir, filename)

        # Check if the file is an image
        if not os.path.isfile(source_path):
            continue

        try:
            # Open and resize the image
            with Image.open(source_path) as img:
                resized_img = img.resize((224, 224), Image.Resampling.LANCZOS)

                # Save the resized image to the target directory
                target_path = os.path.join(target_dir, filename)
                resized_img.save(target_path)

                print(f"Resized and saved: {target_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Define source and target directories
    source_directory = input("Enter the source directory path: ").strip()
    target_directory = input("Enter the target directory path: ").strip()

    # Resize images
    resize_images_to_224(source_directory, target_directory)
