# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-25

""""Creates eda table for the pre-processed training data from the Bank Marketing Predicion project 
(from http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip).
Saves the plots as a png file.
Usage: eda_class_imbalance.py <train> <out_filepath> 


Options:
  <train>             pd.DataFrame with training data set (this is a required option)
  <out_filepath>      The directory and filename where output file will be saved. File extension must be .csv
 (this is a required option)
""" 
import pandas as pd
from docopt import docopt
opt = docopt(__doc__)

def table_class_imbalance(df, path):
    """
    Saves a .csv file summary table with calculated percentage of examples by class to highlight class imbalance in the given path.

    Parameters
    ----------
    df : pd.DataFrame
        data
    path : string
        the directory and filename where output file will be saved
        
    Returns
    -------
    None 
    
    """
    try:
      summary = pd.DataFrame(df.groupby('y').size().reset_index(name='count'))
      summary["perc"]=(summary["count"]/summary["count"].sum()).round(3)
      summary.columns = ["Customer Response","Count","Percentage"]
      summary.to_csv(path)
    except Exception as ex:
        print("Something went wrong2!")

def main(data, path):
    """
    Saves boxplot figure in <out_filepath>

    Parameters
    ----------
    df : pd.DataFrame
        data
    path : string
        the directory where output file will be saved

    Returns
    -------
    None
  """
    try:
        df = pd.read_csv(data)
        table_class_imbalance(df, path)
        print("Saved " + path)
    except Exception as ex:
        print("Something went wrong!")


if __name__ == "__main__":
    main(opt["<train>"], opt["<out_filepath>"])
