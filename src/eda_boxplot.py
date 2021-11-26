# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-25

""""Creates eda plot for the pre-processed training data from the Bank Marketing Predicion project 
(from http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip).
Saves the plots as a png file.
Usage: eda_boxplot.py <train> <out_filepath>


Options:
  <train>           The directory and filename where the training data set is located. Must be a .csv file.
  <out_filepath>    The directory and filename where output file will be saved.  Must be a .png file.
""" 
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import altair_saver
from docopt import docopt
opt = docopt(__doc__)

def boxplot_by_age(df, path):
    """
    Saves boxplot_by_age.png in <path>

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
    (
        alt.Chart(df, title="Distribution of Customer Response by Age")
        .mark_boxplot()
        .encode(
            x=alt.X("age", scale=alt.Scale(zero=False), title="Age"),
            y=alt.Y("y", title="Customer response"),
            color=alt.Color("y", legend=None),
            size="count()",
        )
    ).properties(width=350, height=150).save(
        path, scale_factor=3
    )
  

def main(data, path):
    """
    Creates a boxplot figure with <data> and saves it in <out_filepath>

    Parameters
    ----------
    data : string
        the directory and filename where the training data set is located. Must be a .csv file.
    path : string
        the directory where output file will be saved

    Returns
    -------
    None
  """
    try:
        df = pd.read_csv(data)
        boxplot_by_age(df, path)
        print("Saved " + path)
    except Exception as ex:
        print("Something went wrong!")


if __name__ == "__main__":
    main(opt["<train>"], opt["<out_filepath>"])
