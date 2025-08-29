import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader():

    def __init__(self, data_file_path:str):
        self.__data = pd.read_csv(data_file_path)