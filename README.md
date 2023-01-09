# LSTM Model for Predicting COVID-19 Cases in Malaysia
This project aims to use a Long Short-Term Memory (LSTM) neural network to predict the number of new COVID-19 cases in Malaysia. The model is trained on a dataset of daily cases from January 2020 to the present day.

# Badges

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

In this code there will be 2 sections, where 1st part is loading the data from "cases_malaysia_train.csv" while 2nd part of the code is loading the data from "cases_malaysia_test.csv".

DATA DOCUMENTATION PART 1 (Explanation for each section number)\n

    1. This is where we loading the data from Datasets folder. For this 1st part of the coding, we will be using the "cases_malaysia_train.csv" file.\n

    2. This is where we do data inspection. Data inspection will be checking the data using head(), tail(), info() and describe()\n

    3. Data cleaning 
        After we identify the suspect that cause the dataset is not complete, we then proceed with data cleaning.\n
        Checking the graph we found out that ther is some gap.\n

        3.1 Replacing all the Not A Number (NAN) to numeric using interpolation approach.
        3.2 Check the graph and make sure the graph has complete data.\n

    4. Features Selection
        this is where we use cases_new as our selection for prediction. \n\

    5. Data Preprocessing
        using MinMaxScaler to scale our features\n

        5.1 Create an empty list for X and y with a win_size  = 30
        5.2 We then train test split the datasets for model development\n

    6. Model Development \n

        6.1 Creat a Tensorboard function for further analysis.
        6.2 Create the model using LSTM layers of = 64, and a Dropout at  = 0.3. End the model with an output of 1 with an activation of Relu
        6.3 Run the model with epochs = 10, random_state = 123, shuffle = True

This is the end for part1 Documentation



Ackownldegemetn
https://github.com/MoH-Malaysia/covid19-public
