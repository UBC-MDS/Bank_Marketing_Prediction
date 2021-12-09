# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-25

""""Creates eda plots and tables for the pre-processed training data from the Bank Marketing Predicion project 
(from http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip).
Saves EDA tables and plots with the given path/file prefix.
Usage: eda.py <train> <out_filepath> 


Options:
  <train>             The directory and filename where the training data set is located. Must be a .csv file.
  <out_filepath>      A path/filename prefix where output files will be saved.
 (this is a required option)
""" 
import os, os.path
import errno
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
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
        the path/file prefix where output file will be saved
        
    Returns
    -------
    None 
    
    """
    try:
      summary = pd.DataFrame(df.groupby('y').size().reset_index(name='count'))
      summary["perc"]=(summary["count"]/summary["count"].sum()).round(3)
      summary.columns = ["Customer Response","Count","Percentage"]
      summary.to_csv(path + "/" + "eda_summary_table.csv")
    except Exception as ex:
        print("Something went wrong creating summary table!")


def barchart_by_marital(df, path):
    """
    Saves barchart figure generated with df in <out_filepath>_barchart_by_marital.png

    Parameters
    ----------
    df : pd.DataFrame
        data
    path : string
        the path/file prefix where output file will be saved

    Returns
    -------
    None
    """
    alt.Chart(
        df,
        title="Distribution of Marital Status of customers who suscribed to new product)",
    ).mark_bar().encode(
        y=alt.Y("marital:O", title="Marital Status"),
        x=alt.X("count:Q", scale=alt.Scale(domain=(0, 2100)))
    ).transform_aggregate(
        count='count()',
        groupby=["marital"]
    ).properties(width=350, height=150
    ).save(
        path + "/" + "eda_barchart_by_marital.png", scale_factor=3
    )

def barchart_by_target(df, path):
    """
    Saves barchart figure generated with df in <out_filepath>_barchart_by_target.png

    Parameters
    ----------
    df : pd.DataFrame
        data
    path : string
        the path/file prefix where output file will be saved

    Returns
    -------
    None
    """
    df['response'] = "Rejected subscription"
    df.loc[df['y']==1,'response'] = "Accepted subscription"
    alt.Chart(
        df,
        title="Distribution of Customer Response",
    ).mark_bar().encode(
        y=alt.Y("response:N", title=""),
        x=alt.X("count:Q", scale=alt.Scale(domain=(0, 30000)), title='Customers contacted by phone')
    ).transform_aggregate(
        count='count()',
        groupby=["response"]
    ).properties(width=400, height=100
    ).save(
        path + "/" + "eda_barchart_by_target.png", scale_factor=3
    )
    
  
def boxplot_by_age(df, path):
    """
    Saves boxplot figure generated with df in <out_filepath>_boxplot_by_age.png

    Parameters
    ----------
    df : pd.DataFrame
        data
    path : string
        the path/file prefix where output file will be saved

    Returns
    -------
    None
    """
    (
        alt.Chart(df, title="Distribution of Customer Response by Age")
        .mark_boxplot()
        .encode(
            x=alt.X("age:Q", scale=alt.Scale(zero=False), title="Age"),
            y=alt.Y("y:N", title="Customer response"),
            color=alt.Color("y:N", legend=None),
            size="count()",
        )
    ).properties(width=350, height=150).save(
        path + "/" + "eda_boxplot_by_age.png", scale_factor=3
    )  


def main(data, path):
    """
    Saves EDA tables and plots generated using <data> using the given path/file prefix <out_filepath>

    Parameters
    ----------
    data : string
        the directory and filename where the training data set is located. Must be a .csv file.
    path : string
        the path/file prefix where output files will be saved

    Returns
    -------
    None
  """
    try:
        mkdir_p(path)
        df = pd.read_csv(data)
        table_class_imbalance(df, path)
        boxplot_by_age(df, path)
        barchart_by_marital(df[df["y"] == 1], path)
        barchart_by_target(df, path)
        print("Saved EDA figures in " + path)
    except Exception as ex:
        print("Something went wrong with the output files!", ex)

def mkdir_p(path):
    """
    Creates a new directory in the given path. If the directory already exists it does nothing.

    Parameters
    ----------
    path : pd.string
        the path of the new directory
        
    Returns
    -------
    None 
    
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

if __name__ == "__main__":
    main(opt["<train>"], opt["<out_filepath>"])
