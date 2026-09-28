# -*- coding: utf-8 -*-
"""

"""

import numpy as np

subject_list = np.array(['HV001_1', 'HV002_1', 'HV003_1', 'HV004_1', 'HV005_1', 'HV006_1', 'HV007_1', 'HV008_1', 'HV009_1', 'HV010_1','HV011_1', 'HV012_1', 'HV0013_1', 'HV014_1', 'HV015_1'])
np.random.seed(42)
subject_list_permute = np.random.permutation(subject_list)
subject_split = np.split(subject_list_permute,3)
print(subject_split)
total_count = len(subject_split)

for index_test in range(total_count):
    group_test = subject_split[index_test]
    group_train_all= []    
    for index_train in range(total_count):
        if index_train != index_test:
          group_train = subject_split[index_train]  
          group_train_all.append(group_train)
    print('group_test:',index_test, group_test)
    print('group_train:', index_test, group_train_all)
          
    

    
    