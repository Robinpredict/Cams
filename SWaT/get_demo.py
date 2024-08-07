# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:31:53 2024

@author: ruobi
"""
import numpy as np
L=5000
data = np.load( "SWaT_train_V3.npy")[:L,:]
np.save( 'SWaT_train_V3.npy',data)

data = np.load( "SWaT_train.npy")[:L,:]
np.save( 'SWaT_train.npy',data)
# L=8000
# test_data = np.load( 'SWaT_test_V1'+'.npy')[:L,:]
# np.save( 'SWaT_test_V1.npy',test_data)

# test_data = np.load( 'SWaT_test_V2'+'.npy')[:L,:]
# np.save('SWaT_test_V2.npy',test_data)

# test_data = np.load( 'SWaT_test_V3'+'.npy')[:L,:]
# np.save('SWaT_test_V3.npy',test_data)

# test_labels = np.load( "SWaT_test_label.npy")[:L]
# np.save('SWaT_test_label.npy',test_labels)