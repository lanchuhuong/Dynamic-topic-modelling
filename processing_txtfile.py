import pandas as pd
import os
import sys
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from pywsd.utils import lemmatize_sentence
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from wordcloud import WordCloud, ImageColorGenerator
import scattertext as st
from matplotlib import pyplot as plt

nltk.download("averaged_perceptron_tagger")

dir_path = os.path.dirname(os.path.abspath("__file__"))
main_data_dir = os.path.join(dir_path, "TXT")


def open_speech(file_path):
    """
    This function opens a file with the correct formatting
    :param file_path:
    :return:
    """

    file = open(file_path, encoding="utf-8-sig")
    data = file.read()

    return data


def remove_line_number(speech):
    """
    removes the line number at the beginning of speech

    Parameters
    ---------
    speech : str
        piece of text
    """

    pattern = "\n|^\d+.*?(\w)"
    speech = re.sub(pattern, "\n\g<1>", speech)
    pattern = "\t"
    speech = re.sub(pattern, "", speech)
    pattern = "\n\n"
    speech = re.sub(pattern, "\n", speech)
    pattern = "^\n *"
    speech = re.sub(pattern, "", speech)

    return speech


if __name__ == "__main__":
    # True --> run preprocessing and save the results, False --> just do the data analysis with your previously saved
    # dataframe file (always have to do a preprocessing run to save the dataframe of course)
    do_preprocessing = True

    if do_preprocessing:
        speeches_df = pd.DataFrame(
            columns=[
                "session_nr",
                "year",
                "country",
                "speech",
            ]
        )

        num_directories = len(next(os.walk(main_data_dir))[1])

        # loop through all directories of the data
        for root, subdirectories, files in tqdm(
            os.walk(main_data_dir), total=num_directories, desc="directory: "
        ):
            # remove all the files starting with '.' (files created by opening a mac directory on a windows PC,
            # so will only do something if you are working on a windows PC
            files_without_dot = [file for file in files if not file.startswith(".")]

            # loop through files and extract data
            for file in tqdm(files_without_dot, desc="files: ", leave=False):
                country, session_nr, year = file.replace(".txt", "").split("_")

                # open a speech with the correct formatting
                speech_data = open_speech(os.path.join(root, file))
                speech_data = remove_line_number(speech_data)

                # append the features to the dataframe
                speeches_df = speeches_df.append(
                    {
                        "session_nr": int(session_nr),
                        "year": int(year),
                        "country": country,
                        "speech": speech_data,
                    },
                    ignore_index=True,
                )
        speeches_df.to_csv("Data/Raw/raw_speeches.csv")
