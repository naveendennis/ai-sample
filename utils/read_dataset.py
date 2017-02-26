import pandas as pd


def read_amazon_csv(dir_path):

    """
    Reading the amazon reviews dataset
    :param dir_path: path to open the function
    :return: feature_list and label_list
    """

    data_list = pd.read_csv(dir_path)
    data_list = pd.np.array(data_list)

    label_list = data_list[:, 2]
    feature_list = data_list[:, 0:2]
    label_list = label_list.astype(str)
    return feature_list, label_list


def read_digits_csv(dir_path, feature_size=64):

    """
    Reading digits dataset and any other normal dataset without special characteristics
    :param dir_path: path of the dataset file
    :param feature_size: defaults to 64
    :return:
    """

    dataset_contents = custom_read_csv(dir_path, feature_size)

    feature_train = dataset_contents[0:dataset_contents.shape[0], 0:feature_size]
    label_train = dataset_contents[0:dataset_contents.shape[0], feature_size]
    return feature_train, label_train


def custom_read_csv(file_name, feature_size):
    with open(file_name, 'r') as f:
        file_contents = f.read().strip()
    f.close()
    import numpy as np
    file_contents = np.array(file_contents.split('\n'))
    dataset_contents = np.array([])
    for each_row in file_contents:
        each_row = str(each_row)
        row_contents = np.array([int(rv) for rv in each_row.split(',')])
        dataset_contents = np.append(dataset_contents, row_contents)
    dataset_contents = dataset_contents.reshape((int(dataset_contents.shape[0]/(feature_size+1)), feature_size+1))
    return dataset_contents