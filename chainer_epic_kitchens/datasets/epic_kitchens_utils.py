import os

from chainer.dataset import download

root = 'pfnet/chainercv/epic_kitchens'


def get_epic_kitchens(year):
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, year)
    return base_path
