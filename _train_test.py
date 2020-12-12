#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Deep Learning Class.

This script contains all deep learning tools to train and test on a image 
classification dataset.

Revision History:
        2020-07-12 (Animesh): Baseline Software.

Example:
        from _train_test import NNTools

"""

#___Import Modules:
import os
import json
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from _datagen import Datagen
from efficientNet.model import EfficientNet


#___Global Variables:
TYPE = 'test'
SETTINGS = 'settings.json'
ODIR = "output/"
SEED = 717


#__Classes:
class NNTools:
    """Neural Network Tool Class.
    
    This class contains all methods to complete whole deep learing session.
    
    """

    def __init__(self, settings=SETTINGS, types=TYPE):
        """Constructor.
        
        Args:
            settings (JSON file): Contains all settings manually provided.
            types (list): Contains settings to determine the session is for
                training or testing.

        """
        
        self.types = types
        self.batch_size = 1
        
        # loads contents fron settings file, sets hyperparameters
        with open(settings) as fp:
            content = json.load(fp)[types]
            
            self.shape = content["shape"]
            self.classes = content["classes"]           
            if types == "train":
                self.epochs = content["epoch"]
                self.batch_size = content["batch"]

        # sets output directory and loads models
        if types == "train":
            self.log = self.set_output()
            self.model = EfficientNet.from_pretrained('efficientnet-b7')
        else:
            self.error = self.set_output()
            self.model = EfficientNet.from_pretrained('efficientnet-b7', 
                                            weights_path='models/epoch.pth')

        # creates data generator and sets random state
        self.datagen = Datagen(shape=self.shape)
        torch.manual_seed(SEED)

        return None


    def set_output(self):
        """Output Manager.
        
        This method checks files and directories for producing output during
        training session and creates them if they don't exist.
        
        Returns:
            log (file): Location of log file to dump results during training 
                session.
            error (file): Location of error file to dump falsely predicted
                image path during testing session

        """

        # checks and creates output directories
        if not os.path.exists(ODIR):
            os.mkdir(ODIR)        
        if not os.path.exists(os.path.join(ODIR,"curves")):
            os.mkdir(os.path.join(ODIR,"curves"))        
        if not os.path.exists(os.path.join(ODIR,"weights")):
            os.mkdir(os.path.join(ODIR,"weights"))

        # checks and creates log file and error file to dump results
        log = os.path.join(ODIR,"result.csv")
        error = os.path.join(ODIR,"error.csv")
        if self.types == "train":
            if os.path.exists(log):
                os.remove(log)
                open(log, 'a').close()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            else:
                open(log, 'a').close()

            return log
        
        else:
            return error


    def train(self, trainset, devset):
        """Mathod to run Training Session.
        
        This method runs the complete training session and produces plots and
        results in every epoch.
        
        Args:
            trainset (pandas dataframe): Contains training data.
            devset (pandas dataframe): Contains validation data.

        """
        
        # loads training dataset
        trainset = pd.read_csv(trainset)
        
        # activates GPU support for model
        model = self.model.cuda()
        
        # sets criterion, optimizer and data generator
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        dataloader = DataLoader(dataset=Datagen(trainset, self.shape), 
                                    batch_size=self.batch_size, shuffle=True)
        
        # initialize result holders
        total_loss = []
        dev_accuracy = []
        epoch_loss = 0.0
        accuracy = 0.0

        # loops over training image set
        for epoch in range(1, self.epochs+1):

            # initialize counters
            batch = 0
            running_loss = 0.0
            start = timeit.default_timer()

            # loops over batches
            for image, label in dataloader:

                # processed image count 
                batch += self.batch_size
                
                # set the gradients to zero to avoid accumulation
                optimizer.zero_grad()
                
                # forward + backward + optimize
                output = model(image.cuda())
                loss = criterion(output, label.cuda())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # print status for every 100 mini-batches
                if batch % 100 == 0:                    
                    stop = timeit.default_timer()
                    print('[{0: 3d}, {1: 5d}] loss: {2: 2.7f} time: {3: 2.3f} dev: {4: 2.0f}'\
                          .format(epoch, batch, running_loss/100, 
                                  stop-start, accuracy))

                    # reinitialize counters
                    epoch_loss = running_loss/100
                    running_loss = 0.0
                    start = timeit.default_timer()

            # accuracy count on dev set
            accuracy = self.test(devset)
            dev_accuracy.append(accuracy)

            # total loss count
            total_loss.append(epoch_loss)
            model_path = 'weights/epoch_%d.pth' % epoch
            self.save_model(mfile=os.path.join(ODIR,model_path))
            
            # plotting loss vs epoch curve, produces log file
            self.plot_result(epoch, total_loss, dev_accuracy)
        
        #show finish message
        print("Training finished!")

        return None


    def test(self, testset, display=False):
        """Mathod to run Testing Session.
        
        This method runs the complete testing session producing results and 
        error report.
        
        Args:
            testset (pandas dataframe): Contains testing data.
            display (boolian): Flag to display result or not.
        
        Returns:
            (float): Accuracy percentage.

        """
        
        # loads training dataset
        testset = pd.read_csv(testset)
        
        # activates GPU support
        model = self.model.cuda()

        # sets data generator
        dataloader = DataLoader(dataset=Datagen(testset, self.shape), 
                                batch_size=self.batch_size, shuffle=False) 

        # initialize result holders
        total_accuracy = 0.0
        true_ones = 0.0
        true_zeros = 0.0
        count = 0
        error = []

        # loops over images
        for image, label in dataloader:

            # processed image count 
            count += self.batch_size

            # produces output and prediction
            output = model(image.cuda())
            _, predicted = torch.max(output.data, 1)

            # calculates accuracy
            total_accuracy += (predicted == label.cuda()).sum().item()
            
            # calculates true positive and false positive
            # produces list of falsely predicted image paths
            if self.types == "test":
                if predicted == label.cuda():
                    if label.detach().numpy()[0] == 1:
                        true_ones += 1
                    else:
                        true_zeros += 1
                else:
                    error.append(testset["image"][count-1])
            
            # print status for every 100 mini-batches
            if display and count%100 == 0:
                print("[{0: 5d}] accuracy: {1: 2.2f}".format(count, total_accuracy*100/count))

        # show end results and creates error CSV file
        if display:
            print("total accuracy = %2.2f" % (total_accuracy*100/len(testset)))            
            print("total true positive = %4.0f" % (true_ones))
            print("total true negative = %4.0f" % (true_zeros))
            pd.DataFrame(error, columns =['image']).to_csv(self.error, index=False)

        return total_accuracy*100/len(testset)


    def plot_result(self, epoch, total_loss, dev_accuracy):
        """Managing Result.
        
        This method produces result with required plots in proper format at 
        each epoch.
        
        Args:
            epoch (int): Indicator of epoch count.
            total loss (float): The accumulated loss.
            dev_accuracy (float): Accuracy percentage on validation data.

        """
        
        # loss vs epoch curve
        plt.figure()
        plt.plot(range(1,epoch+1), total_loss, linewidth = 4)
        plt.title("Training")
        fig_path = ODIR + "/curves/Loss Curve.png"
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(fig_path)
        plt.close()

        # dev accuracy vs epoch curve
        plt.figure()
        plt.plot(range(1,epoch+1), dev_accuracy, linewidth = 4)
        plt.title("Training")
        fig_path = ODIR + "/curves/Accuracy Curve.png"
        plt.xlabel("Epoch")
        plt.ylabel("Dev Accuracy")
        plt.savefig(fig_path)
        plt.close()
        
        # saves accuracy values and show finish message
        content = "{0: 4d},{1: 2.2f},\
            Epoch {0: 4d} - accuracy: {1: 2.2f} - best {2: 4d}\n"\
            .format(epoch, dev_accuracy[epoch-1], np.argmax(dev_accuracy)+1)
        
        # writes in log
        with open(self.log, 'a') as fp:
            fp.write(content)

        return None


    def save_model(self, mfile='weights/model.pth'):
        """Mathod to save Trained Model.
        
        This method saves weights of a trained model.
        
        Args:
            mfile (model file): Model file Location to save the model.

        """
 
        print('Saving Model ')
        torch.save(self.model.state_dict(), mfile)
    
        return None


#                                                                              
# end of file
"""ANI717"""