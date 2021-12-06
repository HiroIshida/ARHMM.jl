import os
import numpy as np
import pickle
from mimic.segmentor import Segmentor
from mimic.file import dump_pickled_data

def load_pickle_6compat(filename):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    except UnicodeDecodeError:
        print('probably cache file was created by 2.x but attempt to load by 3.x')
        with open(filename, 'rb') as f:
            obj = pickle.load(f, encoding='latin1')
    return obj

file_arhmm_segmentation = os.path.expanduser("~/tmp/arhmm_result.pickle")
tmp = load_pickle_6compat(file_arhmm_segmentation)
data = [np.array(e) for e in tmp]
segmentor = Segmentor(data)
dump_pickled_data(segmentor, project_name='dish_demo')
