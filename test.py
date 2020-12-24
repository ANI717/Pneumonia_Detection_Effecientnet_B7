#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Classification Deep Learning Tool.

This script runs testing sessions to classify image. 

Revision History:
        2020-07-12 (Animesh): Baseline Software.

Example:
        $ python test.py

"""


#___Import Modules:
from _train_test import NNTools


#___Global Variables:
TEST_DATA = 'data/val.csv'
SETTINGS = 'settings.json'


#___Main Method:
def main():
    """This is the Main Method.

    This method contains testing session for image classification.

    """

    # creates NNTools object dedicated for running training and tesing
    # completes testing session
    Test = NNTools(settings=SETTINGS)
    Test.test(TEST_DATA, display=True)

    return None


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""