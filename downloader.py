# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-18

'''This script downloads data set from given URL and writes it to local.
This script takes a URL and a local file path.

Usage: downloader.py <url> <path>

Options:
<url>              Data set URL (Required)
<path>             Write path (Required)
'''

import pandas as pd
import numpy as np
from docopt import docopt

opt = docopt(__doc__)
print(opt["<url>"])
