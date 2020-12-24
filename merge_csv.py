#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Merge CSV Files.

This script reads contents from different CSV files and merges them in one CSV
file. 

Revision History:
        2020-07-12 (Animesh): Baseline Software.

Example:
        $ python folder_to_csv.py

"""


#___Import Modules:
import pandas as pd
from sklearn.utils import shuffle


#___Global Variables:
# FILE = ["data/lists/NORMAL/train.csv",
#         "data/lists/PNEUMONIA/train.csv"]
# OFILE = "data/train.csv"

# FILE = ["data/lists/NORMAL/val.csv",
#         "data/lists/PNEUMONIA/val.csv"]
# OFILE = "data/val.csv"

# FILE = ["data/lists/NORMAL/train.csv",
#         "data/lists/Augmented/train.csv",
#         "data/lists/PNEUMONIA/train.csv"]
# OFILE = "data/train_aug.csv"

# FILE = ["data/lists/NORMAL/val.csv",
#         "data/lists/Augmented/val.csv",
#         "data/lists/PNEUMONIA/val.csv"]
# OFILE = "data/val_aug.csv"

FILE = ["data/lists/NORMAL/test.csv",
        "data/lists/PNEUMONIA/test.csv"]
OFILE = "data/test.csv"

SEED = 717


#___Main Method:
def main():
    """This is the Main Method.

    This method contains training and testing session for image classification.

    """
    
    # read contents from CSV files
    content = []
    for ifile in FILE:          
        content.extend([pd.read_csv(ifile)])
    
    # combine contents in one list
    df = content[0]
    for frame in content[1:]:
        df = df.append(frame)
    
    # shuffle contents randomly and write in a CSV file
    df = shuffle(df, random_state=SEED)
    df.to_csv(OFILE, index=False)
    
    return None


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""