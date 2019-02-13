import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir


class OilPalmImages(object):
    def __init__(self, file_directory = None):
        self.file_directory = file_directory



    def read_images(self):

        for f in listdir(self.file_directory):
            file_loc = ''.join([self.file_directory, f])
            file_name_components = re.split('[_.]', f)
            df_f = pd.read_table(file_loc, sep = ",", header = header, names = column_names, index_col = False, error_bad_lines = False)


            pix_val_flat = [x for sets in pix_val for x in sets]
            pix_R_val_flat = [sets[0] for sets in pix_val]
            pix_G_val_flat = [sets[1] for sets in pix_val]
            pix_B_val_flat = [sets[2] for sets in pix_val]
