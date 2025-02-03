path_to_data = './data/exams_part0.hdf5'

import h5py
import numpy as np
with h5py.File(path_to_data, 'r') as f:
    print(f.keys())
    exam_id = f['exam_id'][:]
    print(exam_id)
    print(len(exam_id))
    tracings = f['tracings'][:]
    print(tracings)
    print(tracings.shape)
    
    
import h5py

f = h5py.File(path_to_data, 'r')
# Get ids
traces_ids = np.array(f['exam_id'])
x = f['tracings']
print(x)
print(x.shape)
print(np.array_equal(tracings, x))