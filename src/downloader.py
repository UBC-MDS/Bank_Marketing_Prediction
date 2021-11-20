# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-18

'''This script downloads .tsv.gz format data set from given URL and writes it to local.
This script takes a URL and a local file path.

Usage: downloader.py <url> <path>

Options:
<url>              data set URL (Required)
<path>             Write path (Required)
'''

import pandas as pd
import numpy as np
import os, os.path
import errno
from docopt import docopt

opt = docopt(__doc__)

def main(url, path):
    os.makedirs(path)

    file_name = url.split('/')[-1].split('.gz')[0]
    df = pd.read_csv(url, sep='\t')
    df.to_csv(path+'/'+file_name, sep = '\t', index=False)
    print("Download Is Done")
  
if __name__ == "__main__":
    main(opt["<url>"], opt["<path>"])
