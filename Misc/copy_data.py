import os
import shutil

for (root,dirs,files) in os.walk("/home/rayuga/Documents/DataSet/GAN/minor_data/Enhanced_Optic2/"):
    for f in files:
        source_file=os.path.join('/home/rayuga/Documents/DataSet/GAN/Training_Data-(1024x1024)/Test/1024x1024/',f)
        destination_file=os.path.join('/home/rayuga/Documents/DataSet/GAN/minor_data/custom_data/1024x1024/',f)
        shutil.copy(source_file, destination_file)
