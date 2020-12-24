#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data Generator Class.

This script contains data generator and data parsing tools. 

Revision History:
        2020-07-12 (Animesh): Baseline Software.

Example:
        from _datagen import Datagen

"""


#___Import Modules:
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


#___Global Variables:
SHAPE = [100,100]


#__Classes:
class Datagen(Dataset):
    """Neural Network Data Generator.
    
    This class contains all methods to handle data generation for deep learning
    session.
    
    """

    def __init__(self, ilist=None, shape=SHAPE):
        """Constructor.
        
        Args:
            ilist (list): A list of input images.
            shape (list): A list containing image shape [width, height].

        """
          
        self.transform = transforms.Compose([transforms.ToTensor()])        
        self.ilist = ilist
        self.shape = shape

        return None

    
    def get_image(self, iname):
        """Image to Tensor converter.
        
        This method takes an image and returns as deep learning compatible 
        image tensor with proper transformation.
        
        Args:
            iname (image file path): Image file path as input.
        
        Returns:
            image (image tensor): Transformed image tensor.

        """

        image = cv2.imread(iname)
        
        row, column, _ = image.shape
        if row < column:
            image = cv2.copyMakeBorder(image, 
                                (column-row)//2, 0, 0, 0, cv2.BORDER_CONSTANT)
            image = cv2.copyMakeBorder(image, 
                                0, (column-row)//2, 0, 0, cv2.BORDER_CONSTANT)
        elif row > column:
            image = cv2.copyMakeBorder(image, 
                                0, 0, (row-column)//2, 0, 
                                                        cv2.BORDER_CONSTANT)
            image = cv2.copyMakeBorder(image, 
                                0, 0, 0, (row-column)//2,
                                                        cv2.BORDER_CONSTANT)
        
        image = cv2.resize(image, (self.shape[0], self.shape[1]))

        return self.transform(image)


    def get_label(self, label):
        """Tensor converter.
        
        This method takes an integer and converts it to tensor.
        
        Args:
            label (int): An input integer as label.
        
        Returns:
            label (tensor): label transformed to tensor.

        """

        return torch.from_numpy(np.array(label)).long()

    def __getitem__(self, index):
        """Getitem Method.
        
        This method takes image, and label data from a list and convert them to 
        deep learning compatible tensors with proper transformation.
        
        Args:
            index (int): An integer indicating required data index from 
            provided list.
        
        Returns:
            image (image tensor): Transformed image tensor.
            label (tensor): Label data in tensor form.

        """
        
        return self.get_image(self.ilist["image"][index]), \
                    self.get_label(self.ilist["label"][index])


    def __len__(self):
        """Len Method.
        
        This method returns the length of provided list.

        """

        return len(self.ilist)


#                                                                              
# end of file
"""ANI717"""