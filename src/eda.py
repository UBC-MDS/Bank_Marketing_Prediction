# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-25

""""Creates eda plots for the pre-processed training data from the Bank Marketing Predicion project 
(from http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip).
Saves the plots as a png file.
Usage: eda.py <train> <out_dir>


Options:
  <train>     Takes any value (this is a required option)
  <out_dir>     Takes any value (this is a required option)
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
        the directory where output file will be saved

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
        path + "/boxplot_target_vs_age.png", scale_factor=3
    )


def barchart_by_marital(df, path):
    """
    Saves barchart_by_marital.png in <path>

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
        path + "/barchart_by_marital.png", scale_factor=3
    )
  

def main(data, path):
    """
    Saves barchart_by_marital.png and boxplot_by_age.png in <path>

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
        boxplot_by_age(df, path)
        print("Saved " + path + "/boxplot_target_vs_age.png")
        barchart_by_marital(df[df["y"] == "yes"], path)
        print("Saved " + path + "/barchart_by_marital.png")
    except Exception as ex:
        print("Something went wrong!")


if __name__ == "__main__":
    main(opt["<train>"], opt["<out_dir>"])
