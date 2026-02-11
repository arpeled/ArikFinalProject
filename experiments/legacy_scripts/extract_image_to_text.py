from PIL import Image
import pytesseract

# Load your image
image_path = "/Users/arikpeled/Downloads/Picture1.png"  # Replace with your file path
image = Image.open(image_path)

# Extract text from the image
extracted_text = pytesseract.image_to_string(image)
print(extracted_text)
