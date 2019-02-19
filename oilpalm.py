import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
import sys, os
from sklearn.decomposition import PCA
import matplotlib.image as mpimg


class OilPalmImages(object):
    def __init__(self, file_directory = None, pca_components = None):
        self.file_directory = file_directory
        self.pca_components = pca_components
        self.X_train = None
        self.X_labels = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def read_train_images(self):
        self.X_train = self.read_images(self.file_directory)
        return


    def read_images(self, file_directory):
        f_count = 0
        path, dirs, files = next(os.walk(file_directory))
        file_count = len(files)
        for f in listdir(file_directory):
            print("processing image#",f_count)
            im = Image.open(file_directory+f, 'r')
            img= mpimg.imread(file_directory+f)
            #pix_val = list(img.getdata())
            if f_count == 0:
                width, height = im.size
                # X_R = - np.ones([file_count,width*height])
                # X_G = - np.ones([file_count,width*height])
                # X_B = - np.ones([file_count,width*height])
                #X = - np.ones([file_count,3*width*height])
                X = - np.ones([file_count,width*self.pca_components])
            else:
                #file_loc = ''.join([self.file_directory, f])
                width_f, height_f = im.size
                assert width_f==width,  "ERROR width not equal to rest of images"
                assert height_f==height,  "ERROR height not equal to rest of images"

            img_reduced_mat = self.apply_PCA_onImage(img, num_components=self.pca_components)
            X[f_count,:] = img_reduced_mat.reshape((1,width*self.pca_components))
            # X_R[f_count,:] = [sets[0] for sets in pix_val]
            # X_G[f_count,:] = [sets[1] for sets in pix_val]
            # X_B[f_count,:] = [sets[2] for sets in pix_val]
            f_count +=1
        return X


    # def apply_PCA(X_train):
    #     pca_dims = PCA()
    #     pca_dims.fit(X_train)
    #     cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
    #     d = np.argmax(cumsum >= 0.95) + 1
    #     print('reduced dimension of the training data X after PCA=',d)


    def apply_PCA_onImage(self, img, num_components = None, var = 0.95, show_reconstructed_img = False):
        original_shape = img.shape
        X = img.reshape((original_shape[0],original_shape[1]*3))
        if num_components is None:
            pca = PCA(var)
        else:
            pca = PCA(n_components = num_components)
        img_reduced_mat = pca.fit_transform(X)
        if  show_reconstructed_img:
            print('original dimension of the img', X.shape )
            print('reduced dimension of the img after PCA=',img_reduced_mat.shape)
            img_reconst_mat= pca.inverse_transform(img_reduced_mat)
            img_reconst = img_reconst_mat.reshape(original_shape)
            print("-----image reconstructed from PCA reduced representation----")
            # RGB pixel values need to be int (0,255)
            plt.imshow(img_reconst.astype(int))
            return img_reduced_mat, img_reconst
        return img_reduced_mat

def main():
    train_file_directory = "widsdatathon2019/train_images_less/"
    reduced_image_height = 30
    op_instance = OilPalmImages(file_directory= train_file_directory, pca_components= reduced_image_height)
    op_instance.read_train_images()
    return

if __name__ == "__main__":
	main()
