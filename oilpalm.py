import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
import sys, os
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.image as mpimg


class OilPalmImages(object):
    def __init__(self, file_directory = None, labels_file_name = None, pca_components = None, label_selection_probability = 1, label_selection_method = 'hard'):
        self.file_directory = file_directory
        self.pca_components = pca_components
        self.labels_file_name = labels_file_name
        self.label_selection_probability = label_selection_probability
        self.label_selection_method = label_selection_method
        self.labels_dic = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def read_train_images(self):
        self.X_train, self.Y_train = self.read_images(self.file_directory)
        return


    def read_images(self, file_directory):
        f_count = 0
        file_count = (len([f for f in listdir(file_directory)]))
        Y = - np.ones(file_count)
        for f in listdir(file_directory):
            Y[f_count] = self.get_label_forImage(f)
            #pix_val = list(img.getdata())
            if f_count == 0:
                im = Image.open(file_directory+f, 'r')
                width, height = im.size
                # X_R = - np.ones([file_count,width*height])
                # X_G = - np.ones([file_count,width*height])
                # X_B = - np.ones([file_count,width*height])
                #X = - np.ones([file_count,3*width*height])
                X = - np.ones([file_count,width*self.pca_components])
            #-----process training image only if label accepted----------
            if not(Y[f_count] == -1):
                print("processing training sample#",f_count)
                im = Image.open(file_directory+f, 'r')
                img= mpimg.imread(file_directory+f)
                width_f, height_f = im.size
                assert width_f==width,  "ERROR width not equal to rest of images"
                assert height_f==height,  "ERROR height not equal to rest of images"
                img_reduced_mat = self.apply_PCA_onImage(img, num_components=self.pca_components)
                X[f_count,:] = img_reduced_mat.reshape((1,width*self.pca_components))
                f_count +=1
            else:
                print("training sample rejected")
            # X_R[f_count,:] = [sets[0] for sets in pix_val]
            # X_G[f_count,:] = [sets[1] for sets in pix_val]
            # X_B[f_count,:] = [sets[2] for sets in pix_val]


        # discard unused samples (in hard selection method)
        X = X[0:f_count,:]
        Y = Y[0:f_count]
        return X, Y



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


    def get_labels_from_file(self):
        df_labels = pd.read_csv(self.labels_file_name)
        print("plot the distribution of the score value for each of the classes...")
        g = sns.FacetGrid(df_labels, col="has_oilpalm", size=6, aspect=0.9)
        g.map(plt.hist, "score", bins=12, color="r",alpha=0.5)
        plt.show()
        print("plot the histogram of classes of the training samples...")
        plt.hist(df_labels.has_oilpalm.values)
        plt.xlabel('has_oilpalm class')
        plt.show()
        self.labels_dic  =  df_labels.set_index('image_id').to_dict('index')
        #labels_dic['img_000112017.jpg']['score']
        return

    def get_label_forImage(self, img_file_name):
        img_score = self.labels_dic[img_file_name]['score']
        img_label = self.labels_dic[img_file_name]['has_oilpalm']
        if self.label_selection_method == 'hard':
            if img_score >= self.label_selection_probability:
                return img_label
            else: # Remove sample from training data
                return -1

        else:
            if img_score < self.label_selection_probability:
                return int(not img_label)
            else:
                return img_label

def main():
    train_file_directory = "widsdatathon2019/train_images_less/"
    labels_file_name = 'widsdatathon2019/traininglabels.csv'
    label_selection_probability = 1
    label_selection_method = 'hard' # 'soft'
    reduced_image_height = 30
    op_instance = OilPalmImages(file_directory= train_file_directory, labels_file_name = labels_file_name, pca_components= reduced_image_height, label_selection_probability = label_selection_probability, label_selection_method = label_selection_method)
    op_instance.get_labels_from_file()
    op_instance.read_train_images()

    return

if __name__ == "__main__":
	main()
