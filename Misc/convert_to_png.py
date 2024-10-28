import os
from PIL import Image

def jpg_to_png(jpg_filepath, png_filepath):
  """
  Converts a JPEG image to PNG format.

  Args:
      jpg_filepath (str): Path to the input JPEG image file.
      png_filepath (str): Path to save the output PNG image file.
  """
  try:
    # Open the JPEG image
    img = Image.open(jpg_filepath)

    # Convert the image to RGB mode (required for saving as PNG)
    img = img.convert('RGB')

    # Save the image as PNG
    img.save(png_filepath, format='PNG')

    print(f"Converted {jpg_filepath} to PNG successfully!")
  except FileNotFoundError:
    print(f"Error: File not found - {jpg_filepath}")
  except Exception as e:
    print(f"An error occurred: {e}")

for (root,dirs,files) in os.walk("/home/rayuga/Downloads/256x256/"):
    for f in files:
        #print(f)
        jpg_file = root+'/'+f
        png_file = f'/home/rayuga/Downloads/256x256/png/{f}.png'
        jpg_to_png(jpg_file, png_file)
        