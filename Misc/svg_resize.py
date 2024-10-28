from svglib.svglib import svg2rlg
import svgwrite
import os

for (root,dirs,files) in os.walk("/home/rayuga/Downloads/icons/"):
    for f in files:
        svg=svg2rlg(root+f)

        # Get the original width and height
        width, height = svg.size()

        # Calculate the new dimensions (e.g., scale by 2)
        new_width = 32
        new_height = 32

        new_svg=svgwrite.Drawing(os.path.join("/home/rayuga/Documents/GitHub/Project/Portfolio/Theme1/assets/",f),size=(32,32))

        # Scale all elements in the original drawing
        for element in svg.get_elements():
            element.scale(new_width / width, new_height / height)

        # Add the scaled elements to the new drawing
        new_svg.add(svg.get_elements())

        # Save the resized SVG file
        new_svg.save()