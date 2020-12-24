#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Classification Deep Learning Tool.

This script runs trainingsessions to classify image. 

Revision History:
        2020-07-12 (Animesh): Baseline Software.

Example:
        $ python train.py

"""


#___Import Modules:
from _train_test import NNTools


#___Global Variables:
TRAIN_DATA = 'data/train.csv'
VAL_DATA = 'data/val.csv'
SETTINGS = 'settings.json'


#___Main Method:
def main():
    """This is the Main Method.

    This method contains training session for image classification.

    """

    # creates NNTools object dedicated for running training and tesing
    # completes training session
    Train = NNTools(settings=SETTINGS, types='train')
    Train.train(TRAIN_DATA, VAL_DATA)

    return None


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""