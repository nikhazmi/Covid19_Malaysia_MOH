#%% Import Libaries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf

# Import sklearn 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Import Tensorflow
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Sequential, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model

#%%
"""
DATA DOCUMENTATION PART 1 (Explanation for each section number)
    1. This is where we loading the data from Datasets folder. For this 1st part of the coding, we will be using the "cases_malaysia_train.csv" file.
    2. This is where we do data inspection. Data inspection will be checking the data using head(), tail(), info() and describe()
    3. Data cleaning 
        After we identify the suspect that cause the dataset is not complete, we then proceed with data cleaning.
        Checking the graph we found out that ther is some gap.
        3.1 Replacing all the Not A Number (NAN) to numeric using interpolation approach.
        3.2 Check the graph and make sure the graph has complete data.
    4. Features Selection
        this is where we use cases_new as our selection for prediction.
    5. Data Preprocessing
        using MinMaxScaler to scale our features
        5.1 Create an empty list for X and y with a win_size  = 30
        5.2 We then train test split the datasets for model development
    6. Model Development
        6.1 Creat a Tensorboard function for further analysis.
        6.2 Create the model using LSTM layers of = 64, and a Dropout at  = 0.3. End the model with an output of 1 with an activation of Relu
        6.3 Run the model with epochs = 10, random_state = 123, shuffle = True

This is the end for part1 Documentation
""" 

#%%
#1. Data loading
CSV_PATH = os.path.join(os.getcwd(),"Datasets", "cases_malaysia_train.csv")
df  = pd.read_csv(CSV_PATH)

# %%
#2. Data inspection
df.head(10)
df.tail(10)
df.info()
df.describe().T

# %%
#3. Data Cleaning
df['cases_new'] = pd.to_numeric(df['cases_new'], errors = 'coerce')
print(df.isna().sum())
print(df.info())
plt.figure(figsize=(10,10))
plt.plot(df['cases_new'].values)
plt.show()

# %%
#3.1 Replace nan using interpolation approach
df['cases_new'] = df['cases_new'].interpolate(method = 'polynomial', order = 2)
df.isna().sum()

##3.2 Plot new cases
plt.figure(figsize = (10, 10))
plt.plot(df['cases_new'])
plt.show()

# %%
#4. Features selection
df = df['cases_new'].values

# %%
#5. Data preprocessing
mms = MinMaxScaler()
open = mms.fit_transform(df[::, None])

# %%
#5.1 Create an empty list for x and y with a win size of 30
X = []
y = []
win_size = 30

for i in range(win_size, len(open)):
    X.append(open[i-win_size: i])
    y.append(open[i])

X = np.array(X)
y = np.array(y)

# %%
#5.2 Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.02, shuffle = True, random_state = 123)

# %%
#6. Model Development
#6.1 Tensorboard callback function
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
ts_callback = TensorBoard(log_dir=LOGS_PATH)
es_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)

# %%
#6.2 Create a model
model = Sequential()
model.add(Input(shape = (X_train.shape[1:])))
model.add(LSTM(64, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'relu'))
model.summary()

model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['mse', 'mape'])

#6.3 Run the model
model.fit(X_train, y_train, epochs=10, batch_size=128, callbacks=[es_callback,ts_callback])

#%%
"""
DATA DOCUMENTATION PART 2 (Explanation for each section number)
    7. This is where we loading the data from Datasets folder. For this 1st part of the coding, we will be using the "cases_malaysia_test.csv" file.
        7.1. This is where we do data inspection. Data inspection will be checking the data using head(), tail(), info() and describe()
        7.2. Data cleaning 
            After we identify the suspect that cause the dataset is not complete, we then proceed with data cleaning.
            Checking the graph we found out that ther is some gap.
        7.3 Replacing all the Not A Number (NAN) to numeric using interpolation approach.
        7.4 Check the graph and make sure the graph has complete data.
    8. Concat
        Change both of the datsets to DataFrame type for easy concatenation
        8.1 Concat both datasets which "cases_malaysia_train.csv" and "cases_malaysia_test.csv"
    9. Data Preprocessing
        using MinMaxScaler to scale our features
    10. Create an empty list for X and y with using win_size from PART1 of coding.
        10.1 Predict new cases.
    11. Visualisation
        11.1 Set red and blue line for for prediction values and true values.
        11.2 Label X and y axis.
    12. Print errors
        Print mape and mse.
    13. Architecture Model
        Save the LSTM model architecture.
    14. Model Analysis
        Predict X_test
        14.1 Make performance of the model and the reports
        14.2 Display the reports

This is the end for part1 Documentation
""" 

#%% 
# Test Datasets
# Now we add another datasets from cases_malaysia_test.csv

#7. Test dataset
#7.1 loading test datasets
TEST_CSV_PATH = os.path.join(os.getcwd(),"Datasets", "cases_malaysia_test.csv")
test_df = pd.read_csv(TEST_CSV_PATH)

#7.2 data inspection
test_df.head(10)
test_df.tail(10)
test_df.info()
test_df.describe().T

#7.3 Data cleaning
test_df['cases_new'] = pd.to_numeric(test_df['cases_new'], errors = 'coerce')
print(test_df.isna().sum())
print(test_df.info())
plt.figure(figsize=(10,10))
plt.plot(test_df['cases_new'].values)
plt.show()

#7.4 replace nan using interpolation approach
test_df['cases_new'] = test_df['cases_new'].interpolate(method = 'polynomial', order = 2)
test_df.isna().sum()

#7.5 plot new cases for test_data
plt.figure(figsize = (10, 10))
plt.plot(test_df['cases_new'])
plt.show()

#%%
#8. Change bot dataset to Dataframe for concat
test_df = test_df['cases_new'][::, None]
test_df = pd.DataFrame(test_df)
df = pd.DataFrame(df)

#%%
#8.1 concat both datasets (train and test datasets)
concat = pd.concat((df[0], test_df[0]))

# %%
#9. Using MinMaxScaler for preoprocessing with win_size = 30
concat = concat[len(df[0])-win_size:]
concat = mms.transform(concat[::, None])

# %%
#10. Create an empty list for X test, y test
X_testtest = []
y_testtest = []

for i in range(win_size, len(concat)):
    X_testtest.append(concat[i-win_size:i])
    y_testtest.append(concat[i])

X_testtest = np.array(X_testtest)
y_testtest = np.array(y_testtest)

#10.1 Predict new cases
predicted_cases = model.predict(X_testtest)

# %%
#11. Visualisation
#11.1 Set red as predicted and Blue as y_testtest
plt.figure()
plt.plot(predicted_cases, color = 'red')
plt.plot(y_testtest, color ='blue')
plt.legend(['Predicted', 'Actual'])

#11.2 Label x and y axis
plt.xlabel('Time')
plt.ylabel('Number of Cases')
plt.show()

#%%
#12. Print the values for mape and mse
print(mean_absolute_percentage_error(y_testtest, predicted_cases))
print(mean_squared_error(y_testtest, predicted_cases))

# %%
#13. Plot the architecture of the model
plot_model(model,show_shapes=True,show_layer_names=True)

# %%
#14.  Model Analysis
y_predicted =model.predict(X_test)

#%%
#14.1 Make performance of the model and the reports
y_predicted = np.argmax(y_predicted, axis = 1)
y_test = np.argmax(y_test, axis = 1)

print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)

#14.2 Display the reports
disp = ConfusionMatrixDisplay(cm)
disp.plot()
# %%
