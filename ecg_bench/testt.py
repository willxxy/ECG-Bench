file_path = "./data/cpsc/cpsc_2018/g6/A5739"

import wfdb

signal, fields = wfdb.rdsamp(file_path)

print(signal.shape)
print(fields)

