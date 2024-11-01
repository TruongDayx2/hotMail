from PIL import Image, ImageDraw
import os
import numpy as np



# Load the image

image_path = 'hand.PNG'

image = Image.open(image_path)



# Determine the center and radius

width, height = image.size

center_x, center_y = width // 2, height // 2

radius = center_x  # Radius from the center to the left edge



# Create a circular mask

mask = Image.new("L", (width, height), 0)

draw = ImageDraw.Draw(mask)

draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=255)



# Apply the mask to the image

circular_image = Image.new("RGBA", image.size)

circular_image.paste(image, (0, 0), mask=mask)



# Create directory for saving rotated images if it doesn't already exist
output_dir = 'circle'
os.makedirs(output_dir, exist_ok=True)

circular_image.save(f"{output_dir}/circle4_{0}.png")
# Rotate the circular image from 5° to 355°, incrementing by 5°, and save each rotation
for angle in range(5, 360, 5):
    # Rotate the image
    rotated_image = circular_image.rotate(-angle, expand=True)
    
    # Save the rotated image
    rotated_image.save(f"{output_dir}/circle4_{angle}.png")

output_dir
