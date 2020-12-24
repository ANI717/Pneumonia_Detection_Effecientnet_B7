#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data Augmentation Tool.

This script creates an augmented dataset. 

Revision History:
        2020-07-12 (Animesh): Baseline Software.

Example:
        $ python augmentation.py

"""


#___Import Modules:
import os
import cv2


#___Global Variables:
DIR = "data/ZhangLabData/CellData/chest_xray/train/NORMAL"
ODIR = "data/Augmented_Dataset"


#___Main Method:
def main():
    """This is the Main Method.

    This method creates augmented dataset.

    """
    
    # create output directory if required
    if not os.path.exists(ODIR):
        os.makedirs(ODIR)
    
    # read contents from directory
    content = os.listdir(DIR)
    
    # creates augmented dataset
    for item in content:
        image = cv2.imread(DIR + "/" + item)
        if image.shape[0] < image.shape[1]:
            cv2.imwrite(ODIR + "/top_" + item, cv2.copyMakeBorder(image, 
                                3*(image.shape[1]-image.shape[0])//2, 0, 0, 0, 
                                                        cv2.BORDER_CONSTANT))
            cv2.imwrite(ODIR + "/bottom_" + item, cv2.copyMakeBorder(image, 
                                0, 3*(image.shape[1]-image.shape[0])//2, 0, 0, 
                                                        cv2.BORDER_CONSTANT))
    return None


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""