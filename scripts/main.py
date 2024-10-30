import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('../src'))

from data_loader import load_data

def main(file_path):

    # Load Dataset
    df = load_data(file_path)

    print(df.isnull().sum())


if __name__ == "__main__":
    path = '../data/Brent_Oil_Prices.csv'
    main(file_path=path)
