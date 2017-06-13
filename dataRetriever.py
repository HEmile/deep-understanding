# Running retrieveData(file) will return the the rows from 'file'
# as a list, with the exception of the last column, as the first
# returnable. The second is a list of all the rows with only the 
# last column.
import csv
import random


def retrieveData(file_loc="data/UCI_Credit_Card.csv", randomize=True, make_floats=True):
    x = []
    y = []
    with open(file_loc) as data:
        next(data)  # skip header
        read_data = csv.reader(data)
        for column in read_data:
            if make_floats:
                x.append(list(map(float, column[1:-1])))
                y.append(float(column[-1]))
            else:
                x.append(column[1:-1])
                y.append(column[-1])

    if randomize:
        combined = list(zip(x, y))
        random.shuffle(combined)
        x[:], y[:] = zip(*combined)

    return x, y


def get_data(path_features="data/dengue_features_train.csv", path_labels='data/dengue_labels_train',
             randomize=True, make_floats=True, start_index=4, end_index=-1, label_index=3):
    x = []
    with open(path_features) as data:
        next(data)  # skip header
        read_data = csv.reader(data)
        for column in read_data:
            if make_floats:
                x.append(list(map(float, column[start_index:-end_index])))
            else:
                x.append(column[start_index:end_index])
    y=[]
    with open(path_labels) as data:
        next(data)  # skip header
        read_data = csv.reader(data)
        for column in read_data:
            if make_floats:
                y.append(float(column[label_index]))
            else:
                y.append(column[label_index])
    if randomize:
        combined = list(zip(x, y))
        random.shuffle(combined)
        x[:], y[:] = zip(*combined)

    return x, y