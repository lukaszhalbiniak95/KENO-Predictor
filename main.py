# Author: Lukasz Halbiniak
# BLUEY_V3
# Code for weird things
# Final approach!

# Importing necessary libraries
import torch
from torch.ao.nn.quantized.functional import leaky_relu
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import TensorDataset

import numpy as np
from numpy import save
from numpy import load

import matplotlib.pyplot as plt
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import cg, eigs
from sympy.codegen.ast import float16

import os

# Global variables
ngpu = 0
batch_size = 50
num_epochs = 10000
save_model_batch = 100
learning_rate = 0.0001
epoch_test = 1000

# Binary converter
def binary_converter(x, bits):
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)])

# Preparation of data
def preparing_data_function():
    print("")
    print("Data preparation")

    # Reading file
    TXT_File_Data = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    f = open(dir_path + "\\prep_data\\" + "Input_data" + ".txt", "r")
    for x in f:
        TXT_File_Data.append(x)
    f.close()

    # Creating matrix
    TXT_File_Shuffle = np.zeros((1,len(TXT_File_Data)))
    TXT_File_Day = np.zeros((1,len(TXT_File_Data)))
    TXT_File_Month = np.zeros((1,len(TXT_File_Data)))
    TXT_File_Year = np.zeros((1,len(TXT_File_Data)))
    TXT_File_Hour = np.zeros((1,len(TXT_File_Data)))
    TXT_File_Minute = np.zeros((1,len(TXT_File_Data)))
    TXT_File_Number = np.zeros((21, len(TXT_File_Data)))

    # Writing to matrix
    for y in range(len(TXT_File_Data)):
        TXT_File_Data_Row = TXT_File_Data[y]

        # Schuffle
        TXT_File_Shuffle[0,y] = int(TXT_File_Data_Row[0:7])

        # Day
        if int(TXT_File_Data_Row[8]) == 0:
            TXT_File_Day[0,y] = int(TXT_File_Data_Row[9])
        else:
            TXT_File_Day[0, y] = int(TXT_File_Data_Row[8:10])

        # Month
        if int(TXT_File_Data_Row[11]) == 0:
            TXT_File_Month[0,y] = int(TXT_File_Data_Row[12])
        else:
            TXT_File_Month[0, y] = int(TXT_File_Data_Row[11:13])

        # Year
        TXT_File_Year[0,y] = int(TXT_File_Data_Row[14:18])

        # Hour
        if int(TXT_File_Data_Row[19]) == 0:
            TXT_File_Hour[0,y] = int(TXT_File_Data_Row[20])
        else:
            TXT_File_Hour[0, y] = int(TXT_File_Data_Row[19:21])

        # Minutes

        if int(TXT_File_Data_Row[22]) == 0:
            TXT_File_Minute[0,y] = int(TXT_File_Data_Row[23])
        else:
            TXT_File_Minute[0, y] = int(TXT_File_Data_Row[22:24])

        # Numbers

        for z in range(21):
            From_L = 25 +(z*3)
            if int(TXT_File_Data_Row[From_L])== 0:
                TXT_File_Number[z, y] = int(TXT_File_Data_Row[From_L+1])
            else:
                TXT_File_Number[z, y] = int(TXT_File_Data_Row[From_L : From_L + 2])


    # Converting numbers to bits
    Shuffle_Binary = []
    Day_Binary = []
    Month_Binary = []
    Year_Binary =[]
    Hour_Binary = []
    Minutes_Binary = []
    Multiplier_Binary = []
    Numbers_Binary = np.zeros((70,len(TXT_File_Data)))

    for q in range(len(TXT_File_Data)):
        buff = binary_converter(int(TXT_File_Shuffle[0,q]), 24)
        Shuffle_Binary.append(buff)
        buff = binary_converter(int(TXT_File_Day[0, q]), 8)
        Day_Binary.append(buff)
        buff = binary_converter(int(TXT_File_Month[0, q]), 4)
        Month_Binary.append(buff)
        buff = binary_converter(int(TXT_File_Year[0, q]), 12)
        Year_Binary.append(buff)
        buff = binary_converter(int(TXT_File_Hour[0,q]), 8)
        Hour_Binary.append(buff)
        buff = binary_converter(int(TXT_File_Minute[0,q]), 8)
        Minutes_Binary.append(buff)
        buff = binary_converter(int(TXT_File_Number[20, q]), 4)
        Multiplier_Binary.append(buff)
        for e in range(20):
            buff = TXT_File_Number[e,q] -1
            Numbers_Binary[int(buff),q] = 1

    # Converting to array
    Shuffle_Binary = np.array(Shuffle_Binary)
    Day_Binary = np.array(Day_Binary)
    Month_Binary = np.array(Month_Binary)
    Year_Binary = np.array(Year_Binary)
    Hour_Binary = np.array(Hour_Binary)
    Minutes_Binary = np.array(Minutes_Binary)
    Multiplier_Binary = np.array(Multiplier_Binary)
    Numbers_Binary = np.transpose(Numbers_Binary)

    # Making matrixes for saving
    Buff_zeros = np.zeros((len(TXT_File_Data),6))
    Buff_1 = np.concatenate((Shuffle_Binary, Day_Binary),axis=1)
    Buff_1 = np.concatenate((Buff_1,Month_Binary),axis=1)
    Buff_1 = np.concatenate((Buff_1,Year_Binary),axis=1)
    Buff_1 = np.concatenate((Buff_1,Hour_Binary),axis=1)
    Buff_1 = np.concatenate((Buff_1,Minutes_Binary),axis=1)
    Buff_1 = np.concatenate((Buff_1,Buff_zeros),axis=1)

    Buff_2 = Numbers_Binary

    # Normalization
    for q in range(len(TXT_File_Data)):
        for w in range(70):
            if Buff_1[q,w] == 0:
                Buff_1[q, w] =0
            if Buff_2[q,w] == 0:
                Buff_2[q,w] = 0

    # Creating 3D matrix
    lenght_buff = len(TXT_File_Data)
    HMS_In = 3
    HMS = int(lenght_buff-HMS_In)

    Input_Matrix_NN = np.zeros((HMS,2,HMS_In,70))
    Output_Matrix_NN = np.zeros((HMS,2,HMS_In,70))
    Buff_1 = np.expand_dims(Buff_1, axis=0)
    Buff_1 = np.expand_dims(Buff_1, axis=0)
    Buff_2 = np.expand_dims(Buff_2, axis=0)
    Buff_2 = np.expand_dims(Buff_2, axis=0)

    for k in range(HMS):
        x1_cord = k
        x2_cord = k+HMS_In
        x3_cord = k+1
        x4_cord = k+1+HMS_In
        Input_Matrix_NN[k,0,:,:] = Buff_2[0,0, x1_cord:x2_cord, :]
        Input_Matrix_NN[k, 1, :, :] = Buff_1[0, 0, x1_cord:x2_cord, :]
        Output_Matrix_NN[k,0, :, :] = Buff_2[0,0,x3_cord:x4_cord , :]
        Output_Matrix_NN[k, 1, :, :] = Buff_1[0, 0, x3_cord:x4_cord, :]

    # Generating overfitting data
    Overfitting_Input = Input_Matrix_NN[HMS-3:HMS-2,:,:,:]
    Overfitting_output = Output_Matrix_NN[HMS-3:HMS-2,:,:,:]

    # Generating test data
    Test_data_input = Input_Matrix_NN[HMS-2:HMS-1,:,:,:]
    Test_data_output = Output_Matrix_NN[HMS-2:HMS-1,:,:,:]

    # Removing 10 data from database
    Input_Matrix_NN = Input_Matrix_NN[0:HMS-3,:,:,:]
    Output_Matrix_NN =Output_Matrix_NN[0:HMS-3,:,:,:]

    # Saving data
    print("Saving to file")
    np.save(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Input_matrix_NN" + ".npy", Input_Matrix_NN)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Output_matrix_NN" + ".npy", Output_Matrix_NN)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Overfitting_input" + ".npy", Overfitting_Input)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Overfitting_output" + ".npy", Overfitting_output)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Test_data_input" + ".npy", Test_data_input)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Test_data_output" + ".npy", Test_data_output )

class NN_Linear_def(nn.Module):
    def __init__(self):
        super(NN_Linear_def,self).__init__()
        self.f1 = nn.Conv2d(in_channels=2,out_channels=32,kernel_size=(1,5),stride=1,padding=0,bias=True) #50 2 3 70 / 50 32 3 66
        self.f2 = nn.Tanh()
        self.f3 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=(1,5),stride=1,padding=0,bias=True) # 50 32 3 66 / 50 128 3 62
        self.f4 = nn.Tanh()
        self.f5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(2,5),stride=1,padding=0,bias=True) # 50 128 3 62  / 50 256 2 58
        self.f6 = nn.Tanh()
        self.f7 = nn.BatchNorm2d(128)

        self.l1 = nn.LSTM(input_size=58*2,hidden_size=58*2,num_layers=3,batch_first=True,dropout=0.0)


        self.s1 = nn.Linear(58*2,512)
        self.s2 = nn.Tanh()
        self.s3 = nn.Linear(512, 600)
        self.s4 = nn.Tanh()
        self.s5 = nn.Linear(600, 420)
        self.s6 = nn.Sigmoid()


    def forward(self,input):
        output = self.f1(input)
        output = self.f2(output)
        output = self.f3(output)
        output = self.f4(output)
        output = self.f5(output)
        output = self.f6(output)
        output = self.f7(output)
        output = torch.reshape(output, (output.size(0),output.size(1),-1))

        # LSTM - analiza sekwencyjna

        output, _ = self.l1(output)
        output = output[:, -1, :]  # Ostatnia klatka z LSTM


        output = self.s1(output)
        output = self.s2(output)
        output = self.s3(output)
        output = self.s4(output)
        output = self.s5(output)
        output = self.s6(output)

        output = output.view(output.size(0),2,3,70)

        return output

def NN_Loss_def(NN_output,NN_label):

    adversarial_loss = nn.BCELoss()
    NN_Loss = adversarial_loss(NN_output,NN_label)
    return NN_Loss

def show_results():
    print("")
    print("Showing results")
    print("")

    # Loading PYTORCH
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    dir_path = os.path.dirname(os.path.realpath(__file__))

    NN_Show = NN_Linear_def().to(device)
    NN_load = torch.load(dir_path + "\\trained_model\\NN_model_state" + str(600) + ".bin")
    NN_Show.load_state_dict(NN_load['model_state_dict'])

    # Loading test data
    Overfitting_Input = np.load(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Overfitting_input" + ".npy")
    Overfitting_output = np.load(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Overfitting_output" + ".npy")

    Test_data_input = np.load(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Test_data_input" + ".npy")
    Test_data_output = np.load(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Test_data_output" + ".npy")

    # Creating tensor for testing
    Overfitting_Input_Tensor = Variable(torch.Tensor(Overfitting_Input))
    Test_Input_Tensor = Variable(torch.Tensor(Test_data_input))
    Neural_input_1 = Overfitting_Input_Tensor.to(device)
    Neural_input_2 = Test_Input_Tensor.to(device)

    print("")
    print("Generating samples")
    torch.no_grad()
    generated_results_1 = NN_Show(Neural_input_1)

    torch.no_grad()
    generated_results_2 = NN_Show(Neural_input_2)

    generated_results_1 = generated_results_1.detach()
    generated_results_1 = generated_results_1.numpy()
    generated_results_2 = generated_results_2.detach()
    generated_results_2 = generated_results_2.numpy()

    # Cheking
    for w in range(10):
        for q in range(70):
            if generated_results_1[0,0,w,q]>0.1:
                generated_results_1[0,0,w, q] = 1
            elif generated_results_1[0,0,w,q]<0.1:
                generated_results_1[0,0,w, q] = 0
    for w in range(10):
        for q in range(70):
            if generated_results_2[0,0,w,q]>0.5:
                generated_results_2[0,0,w, q] = 1
            elif generated_results_2[0,0,w,q]<0.5:
                generated_results_2[0,0,w, q] = 0

    results_1 = np.zeros((10,1))
    results_3 = np.zeros((10,1))
    results_2 = np.zeros((10,1))
    results_4 = np.zeros((10,1))
    for w in range(10):
        for q in range(70):
            if generated_results_1[0,0,w,q] == Overfitting_output[0,0,w,q] and generated_results_1[0,0,w, q] == 0:
                results_1[w,0] = results_1[w,0] + 1
            if generated_results_1[0,0,w, q] == Overfitting_output[0,0,w, q] and generated_results_1[0,0,w, q] == 1:
                results_3[w,0] = results_3[w,0] +1
    for w in range(10):
        for q in range(70):
            if generated_results_2[0,0,w, q] == Test_data_output[0,0,w, q] and generated_results_2[0,0,w, q] == 0:
                results_2[w,0] = results_2[w,0] + 1
            if generated_results_2[0,0,w, q] == Test_data_output[0,0,w, q] and generated_results_2[0,0,w, q] == 1:
                results_4[w,0] = results_4[w,0] +1

    print("")

def learning_nn_beg():
    print("")
    print("NN learning algorithm")
    print("")

    # Overfitting
    Overfit_trigger = 10
    print("Initialize PYTORCH")
    # Launching torch
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Loading data from file for learning
    print("")
    print("Loading learning data")
    Neural_Input_NN = np.load(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Input_matrix_NN" + ".npy")
    Neural_Output_NN = np.load(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Output_matrix_NN" + ".npy")

    # Overfitting data
    Overfitting_Input = np.load(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Overfitting_input" + ".npy")
    Overfitting_output = np.load(os.path.dirname(os.path.realpath(__file__)) + "\\prep_data\\" + "Overfitting_output" + ".npy")
    Overfitting_Input_Tensor = Variable(torch.Tensor(Overfitting_Input))

    # Creating tensor for learning
    Neural_input = Variable(torch.Tensor(Neural_Input_NN))
    Neural_output = Variable(torch.Tensor(Neural_Output_NN))
    dataset = torch.utils.data.TensorDataset(Neural_input, Neural_output)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Data loaded")

    # Neural network initialization
    NN_Linear_Network = NN_Linear_def().to(device)
    print("")
    print("NN neural network dimensions")
    print(NN_Linear_Network)

    # Optimizer initializing
    print("")
    print("Creating optimilizer")
    NN_optimizer = optim.Adam(NN_Linear_Network.parameters(), lr=learning_rate,weight_decay=0)

    # Main learning function
    for epoch in range(1,num_epochs +1):
        NN_loss_main = []

        for index,(buff_1,buff_2) in enumerate(dataloader):
            # Cleaning gradient
            NN_optimizer.zero_grad()

            # Learning algortihm
            NN_Input_Var = buff_1.to(device)
            NN_Output_Var = NN_Linear_Network(NN_Input_Var)
            NN_real_loss = NN_Loss_def(NN_Output_Var,buff_2)
            NN_real_loss.backward()
            NN_loss_main = NN_real_loss
            NN_optimizer.step()

            # Plotting and saving
            print('[%d/%d][%d/%d]\tLoss_NN: %.8f\t'
                  % (epoch, num_epochs, index, len(dataloader)-1,
                     NN_real_loss))
            f = open(dir_path + '\\trained_model\\Results.txt', 'a')
            f.write('[%d/%d][%d/%d]\tLoss_NN: %.8f\t'
                  % (epoch, num_epochs, index, len(dataloader)-1,
                     NN_real_loss))
            f.write('\n')
            f.close()

        # Overfitting
        Overfitting_Data = Overfitting_Input_Tensor.to(device)
        torch.no_grad()
        Generated_Overfitting = NN_Linear_Network(Overfitting_Data)
        Generated_Overfitting = Generated_Overfitting.detach()
        Generated_Overfitting = Generated_Overfitting.numpy()
        for w in range(3):
            for q in range(70):
                if Generated_Overfitting[0,0, w, q] > 0.7:
                    Generated_Overfitting[0,0,w, q] = 1
                elif Generated_Overfitting[0,0,w, q] < 0.3:
                    Generated_Overfitting[0,0,w, q] = 0
        results_1 = 0
        results_2 = 0
        for w in range(3):
            for q in range(70):
                if Generated_Overfitting[0,0,w, q] == Overfitting_output[0,0,w, q]:
                    results_1 = results_1 + 1
                else:
                    results_1 = results_1 + 0
                if Generated_Overfitting[0,0,w, q] == Overfitting_output[0,0,w, q] and Generated_Overfitting[0,0,w, q] == 1:
                    results_2 = results_2 +1

        if Overfit_trigger < results_1:
            print("Done")
            torch.save({
                'epoch': epoch,
                'model_state_dict': NN_Linear_Network.state_dict(),
                'optimizer_state_dict': NN_optimizer.state_dict(),
                'loss': NN_loss_main,
            }, dir_path + "\\overfit_model\\NN_model_state" + str(epoch) + ".bin")
            Overfit_trigger = results_1
        else:
            print("No overfitting")

        # After a some batch saving model
        if epoch % save_model_batch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': NN_Linear_Network.state_dict(),
                'optimizer_state_dict': NN_optimizer.state_dict(),
                'loss': NN_loss_main,
            }, dir_path + "\\trained_model\\NN_model_state" + str(epoch) + ".bin")


# Main function
def main():

    print("MIMO Bluey V2")
    print("Author: Lukasz Halbiniak")
    mode = 0

    while 1 == 1:
        print("")
        print("Please type what do you want to do (input variable 1, 2... n):")
        print("1. Prepare samples from 'training_data folder")
        print("2. Learning neural network from beggining")
        print("3. Show results")

        mode = input()
        print("You choose mode: " + mode)
        if mode == "1":
            preparing_data_function()
            print("Finish work")
        elif mode == "2":
            learning_nn_beg()
            print("Finish work")
        elif mode == "3":
            show_results()
            print("Finish work")
        else:
            print("Wrong input. Try again")

# Definition of main function
if __name__ == '__main__':
    main()

