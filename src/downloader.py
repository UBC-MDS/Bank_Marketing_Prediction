# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-18

'''This script downloads .tsv.gz format data set from given URL and writes it to local.
This script takes a URL and a local file path.

Usage: downloader.py <url> [--path=PATH]

Options:
<url>                       data set URL (Required)
--path=PATH                 Write path (Optional) [default: Current]
'''

import pandas as pd
import numpy as np
import os, os.path
import errno
import requests
import zipfile
from docopt import docopt

opt = docopt(__doc__)

def main(url, path):
    mkdir_p(path)
    r = requests.get(url)
    filename = path+"/"+url.split('/')[-1] if path != "Current" else url.split('/')[-1]
    print(filename)
    with open(filename,'wb') as output_file:
        output_file.write(r.content)
 
    print('Download Completed!!!')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
  
if __name__ == "__main__":
    main(opt["<url>"], opt["--path"])
