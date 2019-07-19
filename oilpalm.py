import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
import sys, os
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import tensorflow as tf
import random

class OilPalmImages(object):
    """
    This class is used to load and pre-process training images and read the labels from csv file (containing crowd sourced labels with a score)
    """
    def __init__(self, file_directory = None, labels_file_name = None, apply_pca = True, pca_components = None, label_selection_probability = 1, label_selection_method = 'hard', max_samples_class0 = 15000, max_samples_class1 =15000):
        self.file_directory = file_directory
        self.pca_components = pca_components
        self.labels_file_name = labels_file_name
        self.label_selection_probability = label_selection_probability
        self.label_selection_method = label_selection_method
        self.max_samples_class0 = max_samples_class0
        self.max_samples_class1 = max_samples_class1
        self.apply_pca = apply_pca
        self.labels_dic = None
        self.X_train = None
        self.Y_train = None
        self.X_train_split = None
        self.Y_train_split = None
        self.X_test = None
        self.Y_test = None
        self.num_features = None
        self.width = None
        self.height = None

    def read_train_images(self):
        self.X_train, self.Y_train = self.read_images(self.file_directory, self.max_samples_class0, self.max_samples_class1)
        return


    def read_images(self, file_directory, max_samples_class0 , max_samples_class1 ):
        f_count = 0
        class_0_count = 0
        class_1_count = 0
        file_count = (len([f for f in listdir(file_directory)]))
        Y = - np.ones(file_count)
        for f in listdir(file_directory):
            Y[f_count] = self.get_label_forImage(f)
            #pix_val = list(img.getdata())
            if f_count == 0:
                im = Image.open(file_directory+f, 'r')
                self.width, self.height = im.size
                # X_R = - np.ones([file_count,width*height])
                # X_G = - np.ones([file_count,width*height])
                # X_B = - np.ones([file_count,width*height])
                #X = - np.ones([file_count,3*width*height])
                if self.apply_pca:
                    X = - np.ones([file_count,self.width*self.pca_components])
                    self.num_features = self.width*self.pca_components
                else:
                    X = - np.ones([file_count,3*self.width*self.height])
                    self.num_features = self.width*3*self.height


            #-----process training image only if label accepted----------
            if not(Y[f_count] == -1):
                # increment class counter
                if (Y[f_count] == 0):
                    class_0_count +=1
                else:
                    class_1_count +=1
                # only if current counter have not reached maximum process image
                if ((Y[f_count] == 0) and class_0_count<= max_samples_class0) or ((Y[f_count] == 1) and class_1_count<= max_samples_class1):
                    print("processing training sample# ",f_count, 'from class',Y[f_count] )
                    im = Image.open(file_directory+f, 'r')
                    img= mpimg.imread(file_directory+f)
                    width_f, height_f = im.size
                    assert width_f==self.width,  "ERROR width not equal to rest of images"
                    assert height_f==self.height,  "ERROR height not equal to rest of images"
                    if self.apply_pca:
                        img_reduced_mat = self.apply_PCA_onImage(img, num_components=self.pca_components)
                        X[f_count,:] = img_reduced_mat.reshape((1,self.width*self.pca_components))
                    else:
                        pix_val = list(im.getdata())
                        X[f_count,:] = np.asarray([x for sets in pix_val for x in sets])
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

    def get_labels_dataframe(self):
        labels_df = pd.DataFrame()
        labels_df['img_filename'] = self.labels_dic.keys()
        labels_df['img_label'] = list(map(lambda x: self.get_label_forImage(x), self.labels_dic.keys()))

        plt.hist(labels_df.img_label.values)
        plt.xlabel('has_oilpalm class')
        plt.show()
        labels_df['img_label'] = labels_df['img_label'].astype(str)
        self.labels_df = labels_df;
        return labels_df


    def train_NN(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_train,self.Y_train,train_size = 0.75, test_size = 0.25 , random_state=1)
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
        self.clf.fit(X_train, y_train)
        t = self.clf.predict_proba(X_test)
        y_pred = self.clf.predict(X_test)
        print("test score=",self.clf.score(X_test,y_test))
        plt.hist(y_pred)
        plt.show()
        cm = confusion_matrix(y_test, y_pred)
        # Show confusion matrix in a separate window
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        #print('Test ROC:', roc_auc_score(y_train, y_pred))
        self.plotAUC(self.clf,X_test, y_test)
        plt.show()



    def plotAUC(self, model, x_test, y_test):
        preds = model.predict_proba(x_test)[:,1]
        fpr, tpr, _ = metrics.roc_curve(y_test, preds)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        auc_perc=metrics.auc(fpr, tpr)
        plt.text(0.95, 0.01, 'auc value= %.2f' %   auc_perc,verticalalignment='bottom', horizontalalignment='right', fontsize=15)
        plt.xlabel('False Positive Rate',fontsize=17)
        plt.ylabel('True Positive Rate',fontsize=17)
        plt.tick_params(labelsize=15)
        plt.legend(loc="lower right")


    def label_images(self, test_file_directory):
        f_count = 0
        file_count = (len([f for f in listdir(test_file_directory)]))
        test_res = pd.DataFrame()
        test_res['image_id'] = listdir(test_file_directory)
        test_labels = - np.ones(file_count)
        for f in listdir(test_file_directory):
            print("processing test sample# ",f_count )
            im = Image.open(test_file_directory+f, 'r')
            img= mpimg.imread(test_file_directory+f)
            width_f, height_f = im.size
            if self.apply_pca:
                assert width_f==self.width,  "ERROR width not equal to rest of images"
                assert height_f==self.height,  "ERROR height not equal to rest of images"
                img_reduced_mat = self.apply_PCA_onImage(img, num_components=self.pca_components)
                X_features = img_reduced_mat.reshape((1,self.width*self.pca_components))
            else:
                pix_val = list(im.getdata())
                X_features = np.asarray([x for sets in pix_val for x in sets])
            test_labels[f_count] = self.clf.predict(X_features.reshape(1, -1))[0]
            f_count +=1

        test_res['has_oilpalm'] = test_labels
        test_res.to_csv('test_results.csv', index=False)
        return


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
