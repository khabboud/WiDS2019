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



class OilPalmImages(object):
    """
    This class is used to load and pre-process training images and read the labels from csv file (containing crowd sourced labels with a score)
    """
    def __init__(self, file_directory = None, labels_file_name = None, pca_components = None, label_selection_probability = 1, label_selection_method = 'hard', max_samples_class0 = 15000, max_samples_class1 =15000):
        self.file_directory = file_directory
        self.pca_components = pca_components
        self.labels_file_name = labels_file_name
        self.label_selection_probability = label_selection_probability
        self.label_selection_method = label_selection_method
        self.max_samples_class0 = max_samples_class0
        self.max_samples_class1 = max_samples_class1
        self.labels_dic = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.num_features = None

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
                width, height = im.size
                # X_R = - np.ones([file_count,width*height])
                # X_G = - np.ones([file_count,width*height])
                # X_B = - np.ones([file_count,width*height])
                #X = - np.ones([file_count,3*width*height])
                X = - np.ones([file_count,width*self.pca_components])
                self.num_features = width*self.pca_components
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


#
# class OilPalmClassifier(object):
#     """
#     This class can be used to calculate the distance estimation using a ML model as an alternative for pathloss models
#     """
#     def __init__(self, training_data, training_labels):
#         self.X = training_data
#         self.Y = training_labels
#         self.x_train_mean  = None
#         self.x_train_std  = None
#
#         self.num_training_samples = None
#         self.test_results = None
#         self.ML_model = None
#
#
#     def label_images(self, test_file_directory):
#         f_count = 0
#         file_count = (len([f for f in listdir(test_file_directory)]))
#         test_res = pd.DataFrame()
#         test_res['image_id'] = listdir(test_file_directory):
#         test_labels = - np.ones(file_count)
#         for f in listdir(test_file_directory):
#             print("processing test sample# ",f_count )
#             im = Image.open(test_file_directory+f, 'r')
#             img= mpimg.imread(test_file_directory+f)
#             width_f, height_f = im.size
#             assert width_f==width,  "ERROR width not equal to rest of images"
#             assert height_f==height,  "ERROR height not equal to rest of images"
#             img_reduced_mat = self.apply_PCA_onImage(img, num_components=self.pca_components)
#             X_features = img_reduced_mat.reshape((1,width*self.pca_components))
#             test_labels[f_count] = self.clf.predict(X_features.reshape(1, -1))[0]
#             f_count +=1
#
#         test_res['has_oilpalm'] = test_labels
#         test_res.to_csv('test_results.csv')
#         return
#
#     def train_test_data_split(self, X_train, Y_train, normalize_data = True):
#
#
#         self.num_training_samples =len(y_train
#         x_train, x_test, y_train, y_test = train_test_split(X_train,Y_train,train_size = 0.75, test_size = 0.25 , random_state=1)
#
#         #-------------------Normalizing the Data ----------------
#         #-------should use mean/var of training data only -------
#         if normalize_data:
#             x_train_mean = x_train.mean()
#             x_train_std = x_train.std()
#             x_train = (x_train - x_train_mean)/ x_train_std
#             x_test = (x_test - x_train_mean)/ x_train_std
#             self.x_train_mean  = x_train_mean
#             self.x_train_std  = x_train_std
#
#         # X_train, X_test, y_train, y_test = train_test_split(x,y,train_size = 0.75, test_size = 0.25 , random_state=1)
#         return x_train, x_test, y_train, y_test, x_test_rest, num_training_samples
#
#     def plot_correlation_matrix(self, feature_matrix = None):
#         if feature_matrix is None:
#             x = self.train_feature_matrix
#         else:
#             x = feature_matrix
#         #-------------------Correlation Matrix----------------------------------
#         # Compute the correlation matrix
#         corr = x.corr()
#         # Generate a mask for the upper triangle
#         mask = np.zeros_like(corr, dtype=np.bool)
#         mask[np.triu_indices_from(mask)] = True
#         # Set up the matplotlib figure
#         f, ax = plt.subplots(figsize=(11, 9))
#         # Generate a custom diverging colormap
#         cmap = sns.diverging_palette(220, 10, as_cmap=True)
#         # Draw the heatmap with the mask and correct aspect ratio
#         sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
#                     square=True, linewidths=.5, cbar_kws={"shrink": .5})
#         plt.show()
#         return
#
#     def train_test_ML_model(self, categorical_columns = None, x_train = None, y_train = None, x_test = None, y_test = None, x_test_rest = None, model_name = 'linear_regression', kernel = 'linear'):
#         # kernel can be 'linear' , 'rbf', 'poly'
#         # model can be 'linear regression', 'support vector regressor', 'random forest regressor','Bayesian Ridge','Gaussian Process'
#         is_class_data_flag = False
#         if x_train is None:
#             is_class_data_flag = True
#             y_train = self.train_decision_vector.copy()
#             x_train = self.train_feature_matrix.copy()
#             y_test= self.test_decision_vector.copy()
#             x_test= self.test_feature_matrix.copy()
#             x_test_rest = self.test_rest_matrix.copy()
#
#         x_test_without1hot = x_test.copy()
#         feature_importance_exist = False
#
#         # -----------one hot encoding of categorical features---------------------
#         if categorical_columns is not None:
#             for cat in categorical_columns:
#                 x_train[cat] = x_train[cat].astype('category')
#                 x_test[cat] = x_test[cat].astype('category')
#             x_train = pd.get_dummies(x_train)
#             x_test = pd.get_dummies(x_test)
#             hot1_remove_col = x_train.columns[-1]
#             # print('x_train_data_types', x_train.dtypes)
#             # print('x_test_data_types', x_test.dtypes)
#             print('remove one column from the one-hot-encoding, column=', hot1_remove_col)
#             x_train = x_train.drop([hot1_remove_col], axis=1 )
#             x_test = x_test.drop([hot1_remove_col], axis=1 )
#         print('Using ML_model =', model_name)
#         if model_name == 'linear_regression':
#             # Create linear regression object
#             ML_model = linear_model.LinearRegression()
#             feature_importance_exist = True
#             # # The coefficients
#             # print('Coefficients: \n', lin_regr.coef_)
#         elif model_name == 'support vector regressor':
#             ML_model =  SVR(kernel=kernel, C=1e3, gamma=0.1)
#         elif model_name == 'random forest regressor':
#             ML_model = RandomForestRegressor()
#             feature_importance_exist = True
#         elif model_name == 'Bayesian Ridge':
#             ML_model = linear_model.BayesianRidge()
#         elif model_name == 'Gaussian Process':
#             ML_model = gaussian_process.GaussianProcessRegressor()
#
#
#         # Train the model using the training sets
#         ML_model.fit(x_train,y_train)
#         # Make predictions using the testing set
#         distance_y_pred = ML_model.predict(x_test)
#
#         #------ Test the trained model on the test set ------------------
#         # print stats ..........
#         # The mean squared error
#         print("Mean squared error: %.2f"
#               % mean_squared_error(y_test, distance_y_pred))
#         print("RMSE: %.2f"
#               % np.sqrt(mean_squared_error(y_test, distance_y_pred)))
#         # Explained variance score: 1 is perfect prediction
#         print('r2_score: %.2f' % r2_score(y_test, distance_y_pred))
#
#         if feature_importance_exist:
#             if model_name == 'linear_regression':
#                 print("Feature coefficients (importance)...")
#                 # res = pd.DataFrame(ML_model.coef_ ,x_train.columns.values)
#                 # res.plot(kind='bar')
#                 # plt.ylabel('Feature Coefficient')
#                 # plt.show()
#                 print(x_train.columns)
#                 print(type(x_train.columns))
#                 print(ML_model.coef_[0] )
#                 print(type(ML_model.coef_ ))
#                 plt.title("Feature coefficients (importance)...")
#                 plt.bar(x_train.columns.values, ML_model.coef_[0],  color="b", align="center", alpha = 0.5)
#                 plt.xticks(rotation='vertical')
#                 plt.show()
#             else: # i.e., RF model
#                 # -----------Plot the feature importances of the forest
#                 RF_feature_import = ML_model.feature_importances_
#                 plt.figure()
#                 plt.title("Feature coefficients (importance)...")
#                 plt.bar(x_train.columns.values, RF_feature_import,  color="b", align="center", alpha = 0.5)
#                 plt.xticks(rotation='vertical')
#                 plt.show()
#         #---------Combine the test results with the non-feature columns
#         test_results = pd.concat([x_test_without1hot,x_test_rest], axis = 1) # notice that test_rest
#         test_results['distance_estimate_4'] = distance_y_pred
#         test_results = test_results.reset_index(drop=True)
#         if is_class_data_flag:
#             self.test_results = test_results
#             self.ML_model = ML_model
#         return test_results
#     # def cluster_nodes():
#
#
#

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
