# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-25

'''This script split the bank bank-additional-full.csv into the user defined
train split size

Usage: data_preprocessing.py <src> <dest> --test_split=<test_split> [--random_state=<random_state>]

Options:
<src>                                           Path for the raw data
<dest>                                          Path for the destination of processed data
--test_split=<test_split>                       Numeric between 0 and 1 for the train split size (e.g. 0.2 for a 80% train and 20% test split)
--random_state=[<random_state>]                 Integer value for the random state seed

Example:
python src/data_preprocessing.py data/raw/bank-additional/bank-additional-full.csv data\processed --test_split=0.2
'''

import pandas as pd
import os, os.path
import errno
from docopt import docopt
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(path, dest, test_split, random_state):

    #path = './data/raw/bank-additional/bank-additional-full.csv'
    #dest_path = './data/processed/'
    
    mkdir_p(dest)
        
    if os.path.isfile(path):
        if random_state == None:
            random_state = 123
            print('random state not set, default value 123.')
        
        bank_df = pd.read_csv(path, sep=';')
        train_df, test_df = train_test_split(bank_df, test_size = float(test_split), random_state = int(random_state))
        
        train_df.to_csv(dest + 'bank-additional-train.csv', index=False)
        test_df.to_csv(dest + 'bank-additional-test.csv', index=False)
        
        if os.path.isfile(dest + 'bank-additional-train.csv') and os.path.isfile(dest + 'bank-additional-test.csv'):
            print('train and test data created successfully. Files are in /data/src')
        
    else:
        print(f'{path} file does not exist, please run downloader.py to download the file.')
        
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
        
if __name__ == "__main__":
    main(opt["<src>"], opt["<dest>"],opt["--test_split"], opt["--random_state"])
