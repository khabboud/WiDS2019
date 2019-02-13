# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:00:28 2016

@author: khabboud
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#=======================Display data function=====================================================
def displayData(X, sample_width):
    #DISPLAYDATA Display 2D data in a grid
    #   [h, display_array] = DISPLAYDATA(X, sample_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the 
    #   displayed array if requested.    
    # Set sample_width automatically if not passed in
    if not('sample_width' in globals()) or sample_width.size==0:
    	sample_width = int(round(np.sqrt(X.shape[1])));
        
    # Compute rows, cols
    m, n = X.shape;
    sample_height = int(n / sample_width);
    print('Height=',sample_height, 'Width=',sample_width)
    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)));
    display_cols = int(np.ceil(m / display_rows));
    # Between images padding
    pad = 1;
    # Setup blank display
    display_array = - np.ones([pad + display_rows * (sample_height + pad),\
                           pad + display_cols * (sample_width + pad)]);    
    # Copy each example into a patch on the display array
    curr_ex = 0;
    for j in range(0,display_rows):
    	for i in range(0,display_cols):
         if curr_ex >= m:
             break; 
    		# Copy the patch		
    		# Get the max value of the patch
         max_val = max(abs(X[curr_ex, :]));
         range_h=pad + (j) * (sample_height + pad)
         range_w=pad + (i) * (sample_width + pad)
         display_array[range_h:range_h+sample_height, \
                         range_w:range_w+sample_width] = \
                         np.reshape(X[curr_ex, :], [sample_height, sample_width]) / max_val
         curr_ex = curr_ex + 1;
    	if curr_ex >= m:
    		break;     
    # Display Image
    h = plt.imshow(display_array, cmap=plt.get_cmap('gray'), extent=[-1,1,-1,1])
    plt.show()
    return h, display_array
