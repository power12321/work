from __future__ import division, print_function
__version__ = '1.0.0a'
__module_name__ = 'CEEMDAN_LSTM'
print('Importing...', end = '')
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 
from datetime import datetime
from PyEMD import EMD,EEMD,CEEMDAN 
from sampen import sampen2
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.metrics import r2_score # R2
from sklearn.metrics import mean_squared_error # MSE
from sklearn.metrics import mean_absolute_error # MAE
from sklearn.metrics import mean_absolute_percentage_error # MAPE

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.layers import GRU,Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model 
from tensorflow.python.client import device_lib

# An example
def example():
    print("Start your first prediction by following steps:")
    print("check your dataset and use cl.run_example() can directly run the following example around 1000 seconds.")
    print("##################################")
    print("(0) Import:")
    print("    import CEEMDAN_LSTM as cl")
    print("(1) Declare a path for saving files:")
    print("    series = cl.declare_path()")
    print("(2) CEEMDAN decompose:")
    print("    cl.declare_vars(mode='ceemdan') # set decomposition method")
    print("    imfs = cl.emd_decom()")
    print("(3) Sample Entropy:")
    print("    cl.sample_entropy()")
    print("(4) Integrating IMFs:")
    print("    cl.integrate(inte_form=[[0,1],[2,3,4],[5,6,7]]) # form 233")
    print("(5) Forecast:")
    print("    cl.Respective_LSTM()")

# Run the example above
def run_example():
    print('An example of cl.example() is running around 1000 seconds.')
    print('##################################')
    print("\n(1) Declare a path for saving files:")
    print("-------------------------------")
    series = declare_path()
    print("\n(2) CEEMDAN decompose:")
    print("-------------------------------")
    declare_vars(mode='ceemdan') # reset to default value
    imfs = emd_decom()
    print("\n(3) Sample Entropy:")
    print("-------------------------------")
    sample_entropy()
    print("\n(4) Integrating IMFs:")
    print("-------------------------------")
    integrate(inte_form=[[0,1],[2,3,4],[5,6,7]]) # form 233
    print("\n(5) Forecast:")
    print("-------------------------------")
    declare_vars(mode='ceemdan_se',form='233') # declare variables for forecast
    Respective_LSTM() # ceemdan_se233_data.csv

# Run the example above
def run_predict(series,next_pred=True,epochs=1000):
    print('##################################')
    declare_vars(mode='ceemdan') # reset to default value
    df_ceemdan = emd_decom(series=series)
    df_vmd = re_decom(df=df_ceemdan,redecom_mode='vmd',redecom_list=0) 
    global EPOCHS,PATIENCE
    tmp_epochs, tmp_patience = EPOCHS,PATIENCE
    EPOCHS,PATIENCE = epochs,int(epochs/10)
    Ensemble_LSTM(df=df_vmd,show_model=False,next_pred=next_pred)
    EPOCHS,PATIENCE = tmp_epochs, tmp_patience

# Show Tensorflow running device
def show_devices():
    import tensorflow as tf
    print(device_lib.list_local_devices())

# 2.Declare default variables


# The default dataset saving path: D:\\CEEMDAN_LSTM\\
PATH = 'D:\\vscode\\code\\work\\'
# The default figures saving path: D:\\CEEMDAN_LSTM\\figures\\
FIGURE_PATH = PATH+'figures\\'
# The default logs and output saving path: D:\\CEEMDAN_LSTM\\subset\\
LOG_PATH = PATH+'subset\\'
# The default dataset name of a csv file: cl_sample_dataset.csv (must be csv file)
DATASET_NAME = 'guanzhou'
# The default time series dataset. Load from DATASET_NAME or input a pd.Series.
SERIES = None

# Files variables declare functions
# Declare the path
# You can also enter the time series data directly by declare_path(series)
def declare_path(path=PATH,figure_path=FIGURE_PATH,log_path=LOG_PATH,dataset_name=DATASET_NAME,series=SERIES):
    # Check input
    global PATH,FIGURE_PATH,LOG_PATH,DATASET_NAME,SERIES
    for x in ['path','figure_path','log_path','dataset_name']:
        if type(vars()[x])!=str: raise TypeError(x+' should be strings such as D:\\\\CEEMDAN_LSTM\\\\...\\\\.')
    if path == '' or figure_path == '' or log_path == '':
        raise TypeError('PATH should be strings such as D:\\\\CEEMDAN_LSTM\\\\...\\\\.')
    # declare FIGURE_PATH,LOG_PATH if user only inputs PATH or inputs them at different folders

    # Change path
    ori_figure_path, ori_log_path = FIGURE_PATH, LOG_PATH
    if path != PATH: 
        # Fill path if lack like 'PATH=D:\\CEEMDAN_LSTM' to 'PATH=D:\\CEEMDAN_LSTM\\'
        if path[-1] != '\\': path = path + '\\' 
        PATH = path
        FIGURE_PATH,LOG_PATH = PATH+'figures\\',PATH+'subset\\' 
    if figure_path != ori_figure_path: 
        if figure_path[-1] != '\\': figure_path  = figure_path + '\\'
        FIGURE_PATH = figure_path # Separate figure saving path
    if log_path != ori_log_path: 
        if log_path[-1] != '\\': log_path  = log_path + '\\'
        LOG_PATH = log_path # Separate log saving path
    DATASET_NAME,SERIES = dataset_name,series # update variables

    # Check or create a folder for saving 
    print('Saving path: %s'%PATH)
    for p in [PATH,FIGURE_PATH,LOG_PATH]:
        if not os.path.exists(p): os.makedirs(p)

    # Check whether inputting a series 
    if SERIES is not None:
        if not isinstance(series, pd.Series): raise ValueError('The inputting series must be pd.Series.')
        else: 
            print('Get input series named:',str(series.name))
            SERIES = series.sort_index() # sorting
    
    # Load Data for csv file
    else:
        # Check csv file
        if not (os.path.exists(PATH+DATASET_NAME+'.csv')):
            raise ImportError('Dataset is not exists. Please input dataset_name='+DATASET_NAME+' and check it in: '+PATH
                              +'. You can also input a pd.Series directly.')
        else:
            print('Load sample dataset: '+DATASET_NAME+'.csv')
            # Load sample dataset
            df_ETS = pd.read_csv(PATH+DATASET_NAME+'.csv',header=0)
            df_ETS['date'] = pd.to_datetime(df_ETS['date'])

            # Select close data and convert it to time series data 
            if 'date' not in df_ETS.columns or 'close' not in df_ETS.columns: 
                raise ValueError("Please name the date column and the required price column as 'date' and 'close' respectively.")
            SERIES = pd.Series(df_ETS['close'].values,index = df_ETS['date']) #选择收盘价
            SERIES = SERIES.sort_index() # sorting

    # Save the required data to avoid chaanging the original data
    pd.DataFrame.to_csv(SERIES,PATH+DATASET_NAME+'demo_data.csv')

    # Show data plotting
    fig = plt.figure(figsize=(10,4))
    SERIES.plot(label='Original data', color="#0C0C0C") #F27F19 orange #0070C0 blue
    plt.title('Original Dataset Figure')
    plt.xlabel('Time')
    plt.legend()
    plt.savefig(FIGURE_PATH+'Original Dataset Figure.svg', bbox_inches='tight')
    plt.show()

    return SERIES # pd.Series

# Model variables
# Mainly determine the decomposition method 
MODE = 'ceemdan' 
# Integration form only effective after integration
FORM = '' # such as '233' or 233
# The number of previous days related to today
DATE_BACK = 30 
# The length of the days to forecast
PERIODS = 100 
# LSTM epochs
EPOCHS = 100
# Patience of adaptive learning rate and early stop, suggest 1-20
PATIENCE = 10

# Declare model variables
def declare_vars(mode=MODE,form=FORM,data_back=DATE_BACK,periods=PERIODS,epochs=EPOCHS,patience=None):
    print('##################################')
    print('Global Variables')
    print('##################################')
    
    # Change and Check
    global MODE,FORM,DATE_BACK,PERIODS,EPOCHS,PATIENCE
    FORM = str(form)
    MODE,DATE_BACK,PERIODS,EPOCHS = mode.lower(),data_back,periods,epochs
    if patience is None: PATIENCE = int(EPOCHS/10)
    else: PATIENCE = patience
    check_vars()

    # Show
    print('MODE:'+str.upper(MODE))
    print('FORM:'+str(FORM))
    print('DATE_BACK:'+str(DATE_BACK))
    print('PERIODS:'+str(PERIODS))
    print('EPOCHS:'+str(EPOCHS))
    print('PATIENCE:'+str(PATIENCE))

# Check the type of model variables
def check_vars():
    global FORM
    if MODE not in ['emd','eemd','ceemdan','emd_se','eemd_se','ceemdan_se']:
        raise TypeError('MODE should be emd,eemd,ceemdan,emd_se,eemd_se,or ceemdan_se rather than %s.'%str(MODE))
    if not type(FORM) == str:
        raise TypeError('FORM should be strings in digit such as 233 or "233" rather than %s.'%str(FORM))
    if not (type(DATE_BACK) == int and DATE_BACK>0):
        raise TypeError('DATE_BACK should be a positive integer rather than %s.'%str(DATE_BACK))
    if not (type(PERIODS) == int and PERIODS>=0):
        raise TypeError('PERIODS should be a positive integer rather than %s.'%str(PERIODS))
    if not (type(EPOCHS) == int and EPOCHS>0):
        raise TypeError('EPOCHS should be a positive integer rather than %s.'%str(EPOCHS))
    if not (type(PATIENCE) == int and PATIENCE>0):
        raise TypeError('PATIENCE should be a positive integer rather than %s.'%str(PATIENCE))
    if FORM == '' and (MODE in ['emd_se','eemd_se','ceemdan_se']):
        raise ValueError('FORM is not delcared. Please delcare is as form = 233 or "233".')

# Check dataset input a test one or use the default one
def check_dataset(dataset,input_form,no_se=False,use_series=False,uni_nor=False): # uni_nor is using unified normalization method or not
    file_name = ''
    # Change MODE
    global MODE
    if no_se: # change MODE to the MODE without se 
        check_vars()
        if MODE[-3:] == '_se':
            print('MODE is',str.upper(MODE),'now, using %s instead.'%(str.upper(MODE[:-3])))
            MODE = MODE[:-3]
    # Use SERIES as not dataset
    if use_series:
        if SERIES is None: 
            raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
    # Check user input 
    if dataset is not None:  
        if input_form == 'series' :
            if isinstance(dataset, pd.Series):  
                print('Get input pd.Series named:',str(dataset.name))
                input_dataset = dataset.copy(deep=True)
            else: raise ValueError('The inputting series must be pd.Seriesrather than %s.'%type(dataset))
        elif input_form == 'df':
            if isinstance(dataset, pd.DataFrame): 
                print('Get input pd.DataFrame.')
                tmp_sum = None
                if 'sum' in dataset.columns:
                    tmp_sum = dataset['sum']
                    dataset = dataset.drop('sum', axis=1, inplace=False)
                if 'co-imf0' in dataset.columns: col_name = 'co-imf'
                else: col_name = 'imf'
                dataset.columns = [col_name+str(i) for i in range(len(dataset.columns))] # change column names to imf0,imf1,...
                if tmp_sum is not None:  dataset['sum'] = tmp_sum
                input_dataset = dataset.copy(deep=True)
            else: raise ValueError('The inputting df must be pd.DataFrame rather than %s.'%type(dataset))
        else: raise ValueError('Something wrong happen in module %s.'%__name__)
        file_name = 'test_'
    else: # Check default dataset and load
        if input_form == 'series' : # Check SERIES
            if not isinstance(SERIES, pd.Series): 
                raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
            else: input_dataset = SERIES.copy(deep=True)
        elif input_form == 'df':
            check_vars()
            data_path = PATH+MODE+FORM+'_data.csv'
            if not os.path.exists(data_path):
                raise ImportError('Dataset %s does not exist in '%(data_path)+PATH)
            else: input_dataset = pd.read_csv(data_path,header=0,index_col=0)

    # other warnings
    if METHOD == 0 and uni_nor: 
        print('Attention!!! METHOD = 0 means no using the unified normalization method. Declare METHOD by declare_uni_method(method=METHOD)')

    return input_dataset,file_name

# Declare LSTM model variables

# The units of LSTM layers and 3 LSTM layers will set to 4*CELLS, 2*CELLS, CELLS.
CELLS = 32
# Dropout rate of 3 Dropout layers
DROPOUT = 0.2 
# Adam optimizer loss such as 'mse','mae','mape','hinge' refer to https://keras.io/zh/losses/
OPTIMIZER_LOSS = 'mse'
# LSTM training batch_size for parallel computing, suggest 10-100
BATCH_SIZE = 16
# Proportion of validation set to training set, suggest 0-0.2
VALIDATION_SPLIT = 0.1
# Report of the training process, 0 not displayed, 1 detailed, 2 rough
VERBOSE = 0
# In the training process, whether to randomly disorder the training set
SHUFFLE = True


# Declare LSTM variables
def declare_LSTM_vars(cells=CELLS,dropout=DROPOUT,optimizer_loss=OPTIMIZER_LOSS,batch_size=BATCH_SIZE,validation_split=VALIDATION_SPLIT,verbose=VERBOSE,shuffle=SHUFFLE):
    print('##################################')
    print('LSTM Model Variables')
    print('##################################')
    PATIENCE
    # Changepatience=
    global CELLS,DROPOUT,OPTIMIZER_LOSS,BATCH_SIZE,VALIDATION_SPLIT,VERBOSE,SHUFFLE
    CELLS,DROPOUT,OPTIMIZER_LOSS = cells,dropout,optimizer_loss
    BATCH_SIZE,VALIDATION_SPLIT,VERBOSE,SHUFFLE = batch_size,validation_split,verbose,shuffle

    # Check
    if not (type(CELLS) == int and CELLS>0): raise TypeError('CELLS should a positive integer.')
    if not (type(DROPOUT) == float and DROPOUT>0 and DROPOUT<1): raise TypeError('DROPOUT should a number between 0 and 1.')
    if not (type(BATCH_SIZE) == int and BATCH_SIZE>0):
        raise TypeError('BATCH_SIZE should be a positive integer.')
    if not (type(VALIDATION_SPLIT) == float and VALIDATION_SPLIT>0 and VALIDATION_SPLIT<1):
        raise TypeError('VALIDATION_SPLIT should be a number best between 0.1 and 0.4.')
    if VERBOSE not in [0,1,2]:
        raise TypeError('VERBOSE should be 0, 1, or 2. The detail level of the training message')
    if type(SHUFFLE) != bool:
        raise TypeError('SHUFFLE should be True or False.')
    
    # Show
    print('CELLS:'+str(CELLS))
    print('DROPOUT:'+str(DROPOUT))
    print('OPTIMIZER_LOSS:'+str(OPTIMIZER_LOSS))
    print('BATCH_SIZE:'+str(BATCH_SIZE))
    print('VALIDATION_SPLIT:'+str(VALIDATION_SPLIT))
    print('VERBOSE:'+str(VERBOSE))
    print('SHUFFLE:'+str(SHUFFLE))

# Define the Keras model by model = Sequential() with input shape [DATE_BACK,the number of features]
LSTM_MODEL = None 

# Change Kreas model
def declare_LSTM_MODEL(model=LSTM_MODEL):
    print("LSTM_MODEL has changed to be %s and start your forecast."%model)
    global LSTM_MODEL
    LSTM_MODEL = model
            

# Build LSTM model
def LSTM_model(shape):
    if LSTM_MODEL is None:
        model = Sequential()
        model.add(LSTM(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh', return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(LSTM(CELLS*2,activation='tanh',return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(LSTM(CELLS,activation='tanh',return_sequences=False))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1,activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'GRU':
        model = Sequential()
        model.add(GRU(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh', return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(GRU(CELLS*2,activation='tanh',return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(GRU(CELLS,activation='tanh',return_sequences=False))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1,activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'DNN':
        model = Sequential()
        model.add(Dense(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(CELLS*2,activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Flatten())
        model.add(Dense(CELLS,activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1,activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'BPNN':
        model = Sequential()
        model.add(Dense(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Flatten())
        model.add(Dense(1,activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    else: return LSTM_MODEL

# Other variables

# Method for unified normalization only 0,1,2,3
METHOD = 0 

# declare Method for unified normalization
def declare_uni_method(method=None):
    if method not in [0,1,2,3]: raise TypeError('METHOD should be 0,1,2,3.')
    global METHOD
    METHOD = method
    print('Unified normalization method (%d) is start using.'%method)


# 3.Decomposition, Sample entropy, Re-decomposition, and Integration
#==============================================================================================

# EMD decomposition
# -------------------------------
# Decompose adaptively and plot function
# Residue is named the last IMF
# Declare MODE by declare_vars first
def emd_decom(series=None,trials=10,re_decom=False,re_imf=0,draw=True): 
    # Check input
    dataset,file_name = check_dataset(series,input_form='series') # include check_vars()
    series = dataset.values

    # Initialization
    print('%s decomposition is running.'%str.upper(MODE))
    if MODE == 'emd':decom = EMD()
    elif MODE == 'eemd':decom = EEMD()
    elif MODE == 'ceemdan':decom = CEEMDAN()
    else: raise ValueError('MODE must be emd, eemd, ceemdan when EMD decomposing.')

    # Decompose
    decom.trials = trials # Number of the white noise input
    imfs_emd = decom(series)
    imfs_num = np.shape(imfs_emd)[0]

    if draw:
        # Plot original data
        series_index = range(len(series))
        fig = plt.figure(figsize=(16,2*imfs_num))
        plt.subplot(1+imfs_num, 1, 1 )
        plt.plot(series_index, series, color='#0070C0') #F27F19 orange #0070C0 blue
        plt.ylabel('Original data')
    
        # Plot IMFs
        for i in range(imfs_num):
            plt.subplot(1 + imfs_num,1,2 + i)
            plt.plot(series_index, imfs_emd[i, :], color='#F27F19')
            plt.ylabel(str.upper(MODE)+'-IMF'+str(i))
 
        # Save figure
        fig.align_labels()
        plt.tight_layout()
        if file_name == '':
            if (re_decom==False): plt.savefig(FIGURE_PATH+file_name+str.upper(MODE)+' Result.svg', bbox_inches='tight')
            else: plt.savefig(FIGURE_PATH+'IMF'+str(re_imf)+' '+str.upper(MODE)+' Re-decomposition Result.svg', bbox_inches='tight')
        plt.show()
    
    # Save data
    imfs_df = pd.DataFrame(imfs_emd.T)
    imfs_df.columns = ['imf'+str(i) for i in range(imfs_num)]
    if file_name == '':
        if (re_decom==False): 
            pd.DataFrame.to_csv(imfs_df,PATH+file_name+MODE+'_data.csv')
            print(str.upper(MODE)+' finished, check the dataset: ',PATH+file_name+MODE+'_data.csv')

    return imfs_df # pd.DataFrame

# Sample entropy

# You can also enter the imfs_df directly 
def sample_entropy(imfs_df=None): # imfs_df is pd.DataFrame
    df_emd,file_name = check_dataset(imfs_df,input_form='df') # include check_vars()
    if file_name == '': file_name = str.upper(MODE+FORM)
    else: file_name = 'a Test'
    print('Sample entropy of %s is running.'%file_name)  
        
    # Calculate sample entropy with m=1,2 and r=0.1,0.2
    imfs = df_emd.T.values
    sampen = []
    for i in imfs:
        for j in (0.1,0.2):
            sample_entropy = sampen2(list(i),mm=2,r=j,normalize=True)
            sampen.append(sample_entropy)
    
    # Output
    entropy_r1m1,entropy_r1m2,entropy_r2m1,entropy_r2m2 = [],[],[],[]
    for i in range(len(sampen)):
        if (i%2)==0: # r=0.1    
            entropy_r1m1.append(sampen[i][1][1])# m=1
            entropy_r1m2.append(sampen[i][2][1])# m=2
        else: # r=0.2
            entropy_r2m1.append(sampen[i][1][1])# m=1
            entropy_r2m2.append(sampen[i][2][1])# m=2
 
    # Plot     
    fig = plt.figure()
    x = list(range(0,len(imfs),1))
    plt.plot(x,entropy_r1m1,'k:H',label='m=1 r=0.1')
    plt.plot(x,entropy_r2m1,'b:D',label='m=1 r=0.2')
    plt.plot(x,entropy_r1m2,'c:s',label='m=2 r=0.1')
    plt.plot(x,entropy_r2m2,'m:h',label='m=2 r=0.2')
    plt.xlabel('IMFs')
    plt.ylabel('Sample Entropy')
    plt.legend()
    if file_name == '': fig.savefig(FIGURE_PATH+'Sample Entropy of %s IMFs.svg'%(file_name), bbox_inches='tight')
    plt.show()

# Integrate IMFs and Residue

# Residue is the last IMF in dataset
# inte_form defines the IMFs to be integrated such as [[0,1],[2,3,4],[5,6,7]]
def integrate(df=None,inte_form=[[0,1],[2,3,4],[5,6,7]]):
    # Check inte_form
    if type(inte_form)!=list: raise ValueError('inte_form must be a list like [[0,1],[2,3,4],[5,6,7]].')
    # Change to a one-line list for checking duplicates
    check_list = sum(inte_form,[]) # if error, check your inte_form.
    if len(check_list)!=len(set(check_list)): # Check duplicates by set
        raise ValueError('inte_form has repeated IMFs. Please set it again.')
    
    # Check input df and load dataset
    df_emd,file_name = check_dataset(df,input_form='df',no_se=True,use_series=True) # include check_vars()

    # Check inte_form
    if len(check_list) != len(df_emd.columns):
        raise ValueError('inte_form does not match the total number %d of IMFs'%len(df_emd.columns))

    # Integrating (Create Co-IMFs)
    num = len(inte_form) # num is the number of IMFs after integrating, the Co-IMFs
    form = ''
    for i in range(num):
        str_form = ['imf'+str(i) for i in inte_form[i]]
        locals()['co-imf'+str(i)] = pd.Series(df_emd[str_form].sum(axis=1))
        form = form+str(len(inte_form[i])) # name the file and [[0,1],[2,3,4],[4,5,6]] is 233
    print('The Integrating Form:',form)
    
    # Plot original data by SERIES
    fig = plt.figure(figsize=(16,2*num)) 
    plt.subplot(1+num, 1, 1)
    if file_name == '':
        plt.plot(range(len(SERIES)), SERIES, color='#0070C0') #F27F19 orange #0070C0 blue
    else: # if input plot the sum
        df_sum = df_emd.T.sum()
        plt.plot(range(len(df_sum.index)), df_sum.values, color='#0070C0')
    plt.ylabel('Original data')

    # Name the figure of each Co-IMF
    re = ''
    if form=='44': re='Re1-'
    elif form=='323': re='Re2-'
    elif form=='233': re='Ori-'
    elif form=='224': re='Re3-'
    elif form=='2222': re='Re4-'
    elif form=='2123': re='Re5-'
    elif form=='1133': re='Re6-'
        
    # Plot Co-IMFs
    imfs_name, co_imfs = [], []
    for i in range(num):
        plt.subplot(1+num,1,i+2)
        plt.plot(range(len(df_emd.index)), vars()['co-imf'+str(i)], color='#F27F19') #F27F19
        plt.ylabel(re+'Co-IMF'+str(i))
        imfs_name.append('co-imf'+str(i)) # series name of Co-IMFs
        co_imfs.append(vars()['co-imf'+str(i)])
        
    # Save figure
    fig.align_labels()
    plt.tight_layout()
    plt.savefig(FIGURE_PATH+file_name+str.upper(MODE)+' Integration Figure in form '+form+'.svg', bbox_inches='tight')
    plt.show()
    
    # Save Co-IMFs
    df_co_emd = pd.DataFrame(co_imfs).T
    df_co_emd.columns=imfs_name
    if file_name == '': pd.DataFrame.to_csv(df_co_emd,PATH+file_name+MODE+'_se'+form+'_data.csv')
    print('Integration finished, check the dataset: ',PATH+file_name+MODE+'_se'+form+'_data.csv')
    if file_name != '': return df_co_emd

# Re-decomposition

# re_list is the IMF for re-decomposition
def re_decom(df=None,redecom_mode='ceemdan',redecom_list=[0],draw=True,trials=10,imfs_num=10): 
    # Check inputs
    if isinstance(redecom_list, int): redecom_list=[redecom_list] # if redecom_list is int
    if not isinstance(redecom_list, list): 
        raise ValueError('redecom_list must be a list like [0,1] or an integer like 0 or 1.')
    df_emd,file_name = check_dataset(df,input_form='df') # include check_vars()

    # Check redecom_list
    if len(redecom_list) > len(df_emd.columns) or max(redecom_list) > len(df_emd.columns)-1:
        raise  ValueError('redecom_list exceeds the final IMF: '+str(len(df_emd.columns)-1))
    # Check duplicates
    if len(redecom_list)!=len(set(redecom_list)): raise ValueError('redecom_list has repeated IMFs. Please set it again.')
    col_name = df_emd.columns[0][:-1] # get the IMF name, such as co-imf, imf, co-imf-re
    
    # Name the dataset file and change MODE like co-imf0-re0 
    global MODE
    tmp_mode = MODE # for saving 
    redecom_mode = str.lower(redecom_mode)
    if redecom_mode == 'emd': 
        redecom_file_name,MODE = 're','emd'# co-imf0-re0 
    elif redecom_mode == 'eemd': 
        redecom_file_name,MODE = 'ree','eemd'# co-imf0-ree0 
    elif redecom_mode == 'ceemdan': 
        redecom_file_name,MODE = 'rce','ceemdan'# co-imf0-rce0 
    elif redecom_mode == 'vmd':
        redecom_file_name = 'rv' # co-imf0-rv0 
    else: raise ValueError('redecom_mode must be emd, eemd, ceemdan, or vmd.')
    # Re-decompose and create dataset
    redecom_list.sort()
    redecom_imfs_name = '-'+redecom_file_name # new imfs name
    df_redecom = df_emd.copy(deep=True)
    ori_col_names = list(df_emd.columns) # col_names
    df_col_location = 1 # change columns location if re-decompose multiple IMFs
    for i in redecom_list:
        if not isinstance(i, int): raise ValueError('redecom_list must be a list like [0,1] or an integer like 0 or 1.')
        redecom_file_name = redecom_file_name+str(i) # file name
        print('Re-decomposition is running for %s.'%(col_name+str(i)))

        # Re-decompose (figure is saved with name)
        if redecom_mode == 'vmd': df_redecom_ans = vmd_decom(df_emd[col_name+str(i)],re_decom=True,re_imf=i,K=imfs_num,draw=draw)
        else: df_redecom_ans = emd_decom(df_emd[col_name+str(i)],trials=trials,re_decom=True,re_imf=i,draw=draw) # use emd_decom()
        
        df_redecom_ans.columns = [col_name+str(i)+redecom_imfs_name+str(x) for x in range(len(df_redecom_ans.columns))]
        
        # Abandon the original IMF and insert the re-decomposed value
        df_redecom = df_redecom.drop(col_name+str(i), axis=1, inplace=False) # delete original IMF
        df_col_location = i + df_col_location - 1
        ori_col_names.pop(df_col_location) # delete corresponding name
        df_redecom = pd.concat([df_redecom, df_redecom_ans],axis=1)
        
        # Change order for co-imf0-re0 
        ori_col_names[df_col_location:df_col_location] = df_redecom_ans.columns # List of column names in the correct order
        df_col_location = df_col_location + len(df_redecom_ans.columns) - i
        df_redecom = df_redecom.reindex(columns=ori_col_names)

    # Save data and revert MODE
    MODE =  tmp_mode # for saving 
    redecom_file_name = '_'+redecom_file_name # such as _rce0
    if file_name == '':
        print('Re-decomposition finished, check the dataset: ',PATH+file_name+MODE+FORM+redecom_file_name+'_data.csv')
        pd.DataFrame.to_csv(df_redecom,PATH+file_name+MODE+FORM+redecom_file_name+'_data.csv') # ceemdan_se233_rce0_data.csv

    return df_redecom # pd.DataFrame

# VMD # There are some problems in this module

def vmd_decom(series=None,alpha=2000,tau=0,K=5,DC=0,init=1,tol=1e-7,re_decom=True,re_imf=0,draw=True):
    # Check input
    dataset,file_name = check_dataset(series,input_form='series') # include check_vars()

    from vmdpy import VMD  
    # VMD parameters
    #alpha = 2000       # moderate bandwidth constraint  
    #tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
    #K = 3              # 3 modes  
    #DC = 0             # no DC part imposed  
    #init = 1           # initialize omegas uniformly  
    #tol = 1e-7         

    # VMD 
    imfs_vmd, imfs_hat, omega = VMD(series, alpha, tau, K, DC, init, tol)  
    imfs_num = np.shape(imfs_vmd)[0]
    
    if draw:
        # Plot original data
        series_index = range(len(series))
        fig = plt.figure(figsize=(16,2*imfs_num))
        plt.subplot(1+imfs_num, 1, 1 )
        plt.plot(series_index, series, color='#0070C0') #F27F19 orange #0070C0 blue
        plt.ylabel('VMD Original data')
    
        # Plot IMFs
        for i in range(imfs_num):
            plt.subplot(1 + imfs_num,1,2 + i)
            plt.plot(series_index, imfs_vmd[i, :], color='#F27F19')
            plt.ylabel('VMD-IMF'+str(i))

        # Save figure
        fig.align_labels()
        plt.tight_layout()
        if (re_decom==False): plt.savefig(FIGURE_PATH+file_name+'VMD Result.svg', bbox_inches='tight')
        else: plt.savefig(FIGURE_PATH+'IMF'+str(re_imf)+' VMD Re-decomposition Result.svg', bbox_inches='tight')
        plt.show()
    
    # Save data
    imfs_df = pd.DataFrame(imfs_vmd.T)
    imfs_df.columns = ['imf'+str(i) for i in range(imfs_num)]
    if file_name == '':
        if (re_decom==False): 
            pd.DataFrame.to_csv(imfs_df,PATH+file_name+'vmd_data.csv')
            print('VMD finished, check the dataset: ',PATH+file_name+'vmd_data.csv')

    return imfs_df # pd.DataFrame


# 4.LSTM Model Functions

# Model evaluation function

def evl(y_test, y_pred, scale='0 to 1'): # MSE and MAE are different on different scales
    y_test,y_pred = np.array(y_test).ravel(),np.array(y_pred).ravel()
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)*100
    print('##################################')
    print('Model Evaluation with scale of',scale)
    print('##################################')
    print('R2:', r2)
    print('RMSE:', rmse)
    print('MAE:', mae)
    print("MAPE:",mape) # MAPE before normalization may error beacause of negative values
    return [r2,rmse,mae,mape]

# DATE_BACK functions for inputting sets

# Method here is used to determine the Unified normalization, use declare_uni_method(method=METHOD) to declare.
def create_dateback(df,uni=False,ahead=1):
    # Normalize for DataFrame
    if uni and METHOD != 0 and ahead == 1: # Unified normalization
        # Check input and load dataset
        if SERIES is None: raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
        if MODE not in ['emd','eemd','ceemdan']: raise ValueError('MODE must be emd, eemd, ceemdan if you want to try unified normalization method.')
        if not (os.path.exists(PATH+MODE+'_data.csv')): raise ImportError('Dataset %s does not exist in '%(PATH+MODE+'_data.csv'),PATH)
      
        # Load data
        df_emd = pd.read_csv(PATH+MODE+'_data.csv',header=0,index_col=0)
        # Method (1)
        print('##################################')
        if METHOD == 1:
            scalar,min0 = SERIES.max()-SERIES.min(),0 
            print('Unified normalization Method (1):')
        # Method (2)
        elif METHOD == 2:
            scalar,min0 = df_emd.max().max()-df_emd.min().min(),df_emd.min().min()
            print('Unified normalization Method (2):')
        # Method (3)
        elif METHOD == 3:
            scalar,min0 = SERIES.max()-df_emd.min().min(),df_emd.min().min()
            print('Unified normalization Method (3):')

        # Normalize
        df = (df-min0)/scalar
        scalarY = {'scalar':scalar,'min':min0}
        print(df)
        if isinstance(df, pd.DataFrame):
            trainX = df.drop('sum', axis=1, inplace=False)
            trainY = np.array(df['sum']).reshape(-1, 1)
        # Normalize for each IMF in Series
        else:
            trainY = np.array(df.values).reshape(-1, 1)
            trainX = trainY
    else:
        # Normalize without unifying
        if isinstance(df, pd.DataFrame):
            trainX = df.drop('sum', axis=1, inplace=False)
            scalarX = MinMaxScaler(feature_range=(0,1))#sklearn normalize
            trainX = scalarX.fit_transform(trainX)
            trainY = np.array(df['sum']).reshape(-1, 1)
            scalarY = MinMaxScaler(feature_range=(0,1))#sklearn normalize
            trainY = scalarY.fit_transform(trainY)
        # Normalize for each IMF in Series
        else:
            trainY = np.array(df.values).reshape(-1, 1)
            scalarY = MinMaxScaler(feature_range=(0,1))#sklearn normalize
            trainY = scalarY.fit_transform(trainY)
            trainX = trainY
    
    # Create dateback
    dataX, dataY = [], []
    ahead = ahead - 1
    for i in range(len(trainY)-DATE_BACK-ahead):
        dataX.append(np.array(trainX[i:(i+DATE_BACK)]))
        dataY.append(np.array(trainY[i+DATE_BACK+ahead]))
    return np.array(dataX),np.array(dataY),scalarY,np.array(trainX[-DATE_BACK:])

# Plot original data and forecasting data
def plot_all(lstm_type,pred_ans):
    # Check and Change
    if not isinstance(SERIES, pd.Series):
        raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
    pred_ans = pred_ans.ravel()
    series_pred = SERIES.copy(deep=True) # copy original data
    for i in range(PERIODS):
        series_pred[-i-1] = pred_ans[-i-1]

    # Plot
    fig = plt.figure(figsize=(10,4))
    SERIES[-PERIODS*3:].plot(label= 'Original data', color='#0070C0') #F27F19 orange #0070C0 blue
    series_pred[-PERIODS:].plot(label= 'Forecasting data', color='#F27F19')
    plt.xlabel('')
    plt.title(lstm_type+' LSTM forecasting results')
    plt.legend()
    plt.savefig(FIGURE_PATH+lstm_type+' LSTM forecasting results.svg', bbox_inches='tight')
    plt.show()
    return 

# Declare LSTM forecasting function
# Have declared LSTM model variables at Section 0 before

def LSTM_pred(data=None,draw=True,uni=False,show_model=True,train_set=None,next_pred=False,ahead=1):
    # Divide the training and test set
    if train_set is None:
        trainX,trainY,scalarY,next_trainX = create_dateback(data,uni=uni,ahead=ahead)
    else: trainX,trainY,scalarY,next_trainX = train_set[0],train_set[1],train_set[2],train_set[3]
    if uni==True and next_pred==True: raise ValueError('Next pred does not support unified normalization.')

    if PERIODS == 0:
        train_X = trainX
        y_train = trainY
    else:
        x_train,x_test = trainX[:-PERIODS],trainX[-PERIODS:]
        y_train,y_test = trainY[:-PERIODS],trainY[-PERIODS:]
        # Convert to tensor 
        train_X = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        test_X = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Build and train the model
    # print('trainX:\n',train_X[-1:])
    print('\nInput Shape: (%d,%d)\n'%(train_X.shape[1],train_X.shape[2]))
    model = LSTM_model(train_X.shape)
    if show_model: model.summary() # The summary of layers and parameters
    EarlyStop = EarlyStopping(monitor='val_loss',patience=5*PATIENCE,verbose=VERBOSE, mode='auto') # realy stop at small learning rate
    Reduce = ReduceLROnPlateau(monitor='val_loss',patience=PATIENCE,verbose=VERBOSE,mode='auto') # Adaptive learning rate
    history = model.fit(train_X, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                        verbose=VERBOSE, shuffle=SHUFFLE, callbacks=[EarlyStop,Reduce])

    # Plot the model structure
    #plot_model(model,to_file=FIGURE_PATH+'model.png',show_shapes=True)

    # Predict
    if PERIODS != 0:
        pred_test = model.predict(test_X)
        # Evaluate model with scale 0 to 1
        evl(y_test, pred_test) 
    else: pred_test = np.array([])

    if next_pred:# predict tomorrow not in test set
        next_ans = model.predict(next_trainX.reshape((1, trainX.shape[1], trainX.shape[2])))
        pred_test = np.append(pred_test,next_ans)
    pred_test = pred_test.ravel().reshape(-1,1)

    # De-normalize 
    # IMPORTANT!!! It may produce some negative data impact evaluating
    if isinstance(scalarY, MinMaxScaler):
        test_pred = scalarY.inverse_transform(pred_test)
        if PERIODS != 0: test_y = scalarY.inverse_transform(y_test)
    else:     
        test_pred = pred_test*scalarY['scalar']+scalarY['min']
        if PERIODS != 0:test_y = y_test*scalarY['scalar']+scalarY['min']
    
    # Plot 
    if draw and PERIODS != 0:
        # determing the output name of figures
        fig_name = ''
        if isinstance(data,pd.Series): 
            if str(data.name) == 'None': fig_name = 'Series'
            else: fig_name = str(data.name)
        else: fig_name = 'DataFrame'

        # Plot the loss figure
        fig = plt.figure(figsize=(5,2))
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.title(fig_name+' LSTM loss chart')
        plt.savefig(FIGURE_PATH+fig_name+' LSTM loss chart.svg', bbox_inches='tight')
        plt.show()
        
        # Plot observation figures
        fig = plt.figure(figsize=(5,2))
        plt.plot(test_y)
        plt.plot(test_pred)
        plt.title(fig_name+' LSTM forecasting result')
        plt.savefig(FIGURE_PATH+fig_name+' LSTM forecasting result.svg', bbox_inches='tight')
        plt.show()

    return test_pred


# 5.CEEMDAN-LSTM Forecasting Functions
# Please use cl.declare_vars() to determine variables.


# Single LSTM Forecasting without CEEMDAN

# It uses LSTM directly for prediction wiht input_shape=[DATE_BACK,1]
def Single_LSTM(series=None,draw=True,uni=False,show_model=True,next_pred=False,ahead=1):
    print('==============================================================================================')
    print('This is Single LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input series and load dataset
    input_series,file_name = check_dataset(series,input_form='series',uni_nor=uni) # include check_vars()

    # Show the inputting data
    print('Part of Inputting dataset:')
    print(input_series)
    
    # Forecast and save result
    start = time.time()
    test_pred = LSTM_pred(data=input_series,draw=draw,uni=uni,show_model=show_model,next_pred=next_pred,ahead=ahead)
    end = time.time()
    df_pred = pd.DataFrame(test_pred)
    pd.DataFrame.to_csv(df_pred,LOG_PATH+file_name+'single_pred.csv')

    # Evaluate model 
    if draw and file_name == '': plot_all('Single',test_pred[0:PERIODS])  # plot chart to campare
    df_evl = evl(input_series[-PERIODS:].values,test_pred[0:PERIODS],scale='input series') 
    print('Running time: %.3fs'%(end-start))
    df_evl.append(end-start)
    df_evl = pd.DataFrame(df_evl).T #['R2','RMSE','MAE','MAPE','Time']
    pd.DataFrame.to_csv(df_evl,LOG_PATH+file_name+'single_log.csv',index=False,header=0,mode='a') # log record
    print('Single LSTM Forecasting finished, check the logs',LOG_PATH+file_name+'single_log.csv')
    if next_pred: 
        print('##################################')
        print('Today is',input_series[-1:].values,'but predict as',df_pred[-2:-1].values)
        print('Next day is',df_pred[-1:].values)
    if file_name != '': return df_pred

# Ensemble LSTM Forecasting with 3 Co-IMFs

# It uses LSTM directly for prediction wiht input_shape=[DATE_BACK,the number of features]
def Ensemble_LSTM(df=None,draw=True,uni=False,show_model=True,next_pred=False,ahead=1):
    print('==============================================================================================')
    print('This is Ensemble LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input dataset and load 
    input_df,file_name = check_dataset(df,input_form='df',use_series=True,uni_nor=uni) # include check_vars()
    
    # Create ans show the inputting data set
    if file_name == '': input_df['sum'] = SERIES.values # add a column for sum data
    # if input a df, the sum of all columns will be used as the sum price to forecast
    elif 'sum' not in input_df.columns: input_df['sum'] = input_df.T.sum().values 
    print('Part of Inputting dataset:')
    print(input_df)

    # Forecast
    start = time.time()
    test_pred = LSTM_pred(data=input_df,draw=draw,uni=uni,show_model=show_model,next_pred=next_pred,ahead=ahead)
    end = time.time()
    df_pred = pd.DataFrame(test_pred)
    pd.DataFrame.to_csv(df_pred,LOG_PATH+file_name+'ensemble_'+MODE+FORM+'_pred.csv')
    
    # Evaluate model 
    if PERIODS != 0:
        if draw and file_name == '': plot_all('Ensemble',test_pred[0:PERIODS])  # plot chart to campare
        df_evl = evl(input_df['sum'][-PERIODS:].values,test_pred[0:PERIODS],scale='input df') 
        print('Running time: %.3fs'%(end-start))
        df_evl.append(end-start)
        df_evl = pd.DataFrame(df_evl).T #['R2','RMSE','MAE','MAPE','Time']
        if next_pred: 
            print('##################################')
            print('Today is',input_df['sum'][-1:].values,'but predict as',df_pred[-2:-1].values)
            print('Next day is',df_pred[-1:].values)
        pd.DataFrame.to_csv(df_evl,LOG_PATH+file_name+'ensemble_'+MODE+FORM+'_log.csv',index=False,header=0,mode='a') # log record
        print('Ensemble LSTM Forecasting finished, check the logs',LOG_PATH+file_name+'ensemble_'+MODE+FORM+'_log.csv')
    return df_pred

# Respective LSTM Forecasting for each Co-IMF

# It uses LSTM to predict each IMFs respectively input_shape=[DATE_BACK,1]
def Respective_LSTM(df=None,draw=True,uni=False,show_model=True,next_pred=False,ahead=1):
    print('==============================================================================================')
    print('This is Respective LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input dataset and load 
    input_df,file_name = check_dataset(df,input_form='df',use_series=True,uni_nor=uni) # include check_vars()
    data_pred = [] # list for saving results of each Co-IMF
    print('Part of Inputting dataset:')
    print(input_df)
    
    # Forecast
    start = time.time()
    if MODE[-3:]=='_se': col_name = 'co-imf'
    else: col_name = 'imf'
    df_len = len(input_df.columns)
    if 'sum' in input_df.columns: df_len = df_len - 1
    for i in range(df_len):
        print('==============================================================================================')
        print(str.upper(MODE)+'--IMF'+str(i))
        print('==============================================================================================')
        test_pred = LSTM_pred(data=input_df[col_name+str(i)],draw=draw,uni=uni,show_model=show_model,next_pred=next_pred,ahead=ahead)
        data_pred.append(test_pred.ravel())
    end = time.time()

    # Save the forecasting result
    df_pred = pd.DataFrame(data_pred).T
    df_pred.columns = [col_name+str(i) for i in range(len(df_pred.columns))]
    pd.DataFrame.to_csv(df_pred,LOG_PATH+file_name+'respective_'+MODE+FORM+DATASET_NAME+'_pred.csv')

    # Evaluate model 
    if PERIODS != 0:
        res_pred = df_pred.T.sum()
        if draw and file_name == '': plot_all('Respective',res_pred[:PERIODS])  # plot chart to campare
        if file_name == '': input_df['sum'] = SERIES.values # add a column for sum data
        elif 'sum' not in input_df.columns: input_df['sum'] = input_df.T.sum().values 
        df_evl = evl(input_df['sum'][-PERIODS:].values,res_pred[:PERIODS],scale='input df') 
        print('Running time: %.3fs'%(end-start))
        df_evl.append(end-start)
        df_evl = pd.DataFrame(df_evl).T #['R2','RMSE','MAE','MAPE','Time']
        if next_pred: 
            print('##################################')
            print('Today is',input_df['sum'][-1:].values,'but predict as',res_pred[-2:-1].values)
            print('Next day is',res_pred[-1:].values)
        pd.DataFrame.to_csv(df_evl,LOG_PATH+file_name+'respective_'+MODE+FORM+DATASET_NAME+'_log.csv',index=False,header=0,mode='a') # log record
        print('Respective LSTM Forecasting finished, check the logs',LOG_PATH+file_name+'respective_'+MODE+FORM+'_log.csv')
    return df_pred

# Each Multi_pred() takes long time to run around 1000s unless setting the EPOCHS and n.
class HiddenPrints: # used to hide the print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def Multi_pred(df=None,run_times=10,uni_nor=False,single_lstm=False,ensemble_lstm=False,respective_lstm=False,hybrid_lstm=False,redecom=None,ahead=1):
    print('Multiple predictions of '+str.upper(MODE)+FORM+' is running...')
    input_df,file_name = check_dataset(df,input_form='df',use_series=True,uni_nor=uni_nor) # include check_vars()
    if file_name == '': input_series = None
    else: input_series = input_df.T.sum()
    start = time.time()
    with HiddenPrints():
        for i in range(run_times):
            if single_lstm: Single_LSTM(series=input_series,draw=False,uni=uni_nor,ahead=ahead)
            if ensemble_lstm: Ensemble_LSTM(df=df,draw=False,uni=uni_nor,ahead=ahead)
            if respective_lstm: Respective_LSTM(df=df,draw=False,uni=uni_nor,ahead=ahead)
    end = time.time()
    print('Multiple predictions completed, taking %.3fs'%(end-start))
    print('Please check the logs in: '+LOG_PATH)
    
