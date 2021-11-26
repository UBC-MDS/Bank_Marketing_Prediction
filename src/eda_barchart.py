# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-25

""""Creates eda plots for the pre-processed training data from the Bank Marketing Predicion project 
(from http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip).
Saves the plots as a png file.
Usage: eda_barchart.py <train> <out_filepath>


Options:
  <train>     pd.DataFrame with training data set (this is a required option)
  <out_filepath>     The directory and filename where output file will be saved
 (this is a required option)
""" 
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import altair_saver
from docopt import docopt
opt = docopt(__doc__)


def barchart_by_marital(df, path):
    """
    Saves barchart figure in <out_filepath>

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
    (
        alt.Chart(
            df,
            title="Distribution of Marital Status of customers who suscribed to new product)",
        )
        .mark_bar()
        .encode(y=alt.Y("marital", title="Marital Status"), x="count()")
    ).properties(width=350, height=150).save(
        path, scale_factor=3
    )
  

def main(data, path):
    """
    Saves barchart figure in <out_filepath>

    Parameters
    ----------
    df : pd.DataFrame
        data
    path : string
        the directory and filename where the figure will be saved

    Returns
    -------
    None
  """
    try:
        df = pd.read_csv(data)
        barchart_by_marital(df[df["y"] == "yes"], path)
        print("Saved " + path)
    except Exception as ex:
        print("Something went wrong!")


if __name__ == "__main__":
    main(opt["<train>"], opt["<out_filepath>"])
