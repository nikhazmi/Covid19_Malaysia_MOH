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

DATA DOCUMENTATION PART 1.

    1. This is where we loading the data from Datasets folder. For this 1st part of the coding, we will be using the "cases_malaysia_train.csv" file.

    2. This is where we do data inspection. Data inspection will be checking the data using head(), tail(), info() and describe()

    3. Data cleaning 
        After we identify the suspect that cause the dataset is not complete, we then proceed with data cleaning.\n
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

This is the end for part1 of Documentation

DATA DOCUMENTATION PART 2.

    7. This is where we loading the data from Datasets folder. For this 1st part of the coding, we will be using the "cases_malaysia_test.csv" file.\n

        7.1. This is where we do data inspection. Data inspection will be checking the data using head(), tail(), info() and describe()
        7.2. Data cleaning 
            After we identify the suspect that cause the dataset is not complete, we then proceed with data cleaning.
            Checking the graph we found out that ther is some gap.
        7.3 Replacing all the Not A Number (NAN) to numeric using interpolation approach.
        7.4 Check the graph and make sure the graph has complete data.\n

    8. Concat
        Change both of the datsets to DataFrame type for easy concatenation\n

        8.1 Concat both datasets which "cases_malaysia_train.csv" and "cases_malaysia_test.csv"\n

    9. Data Preprocessing
        using MinMaxScaler to scale our features\n

    10. Create an empty list for X and y with using win_size from PART1 of coding.
        10.1 Predict new cases.\n

    11. Visualisation
        11.1 Set red and blue line for for prediction values and true values.
        11.2 Label X and y axis.\n

    12. Print errors
        Print mape and mse.\n

    13. Architecture Model
        Save the LSTM model architecture.\n

    14. Model Analysis
        Predict X_test\n

        14.1 Make performance of the model and the reports
        14.2 Display the reports

This is the end for part2 of Documentation

## Graphs

# Part 1
![Incomplete graph 1](https://user-images.githubusercontent.com/82282919/211273268-347a3f76-8d89-4300-9143-a81d4930e37d.png)
![Complete graph 1](https://user-images.githubusercontent.com/82282919/211273037-4e398a9e-545d-4680-b095-b8a31b5d6aef.png)

# Part 2
![Incomplete graph 2](https://user-images.githubusercontent.com/82282919/211273189-654fa694-b98e-4838-b34d-0dc4883fcc8f.png)
![Complete graph 2](https://user-images.githubusercontent.com/82282919/211273212-f8ff47a5-0725-4a04-a2c2-2ff25cdb9457.png)

# Predict Values VS True Values
![Predict VS True](https://user-images.githubusercontent.com/82282919/211268460-23e35bb5-e052-455c-9bca-374c80c1ce9b.png)

# Architecture of the Model
![model](https://user-images.githubusercontent.com/82282919/211273934-6779151e-6f81-4a29-8c18-d87b3602e2dd.png)


## Acknowledgement
Thank you to the Ministry Of Health (MOH) for providing the COVID-19 case data used in this project. Without their ongoing efforts to track and report on the pandemic, this project would not have been possible. We are grateful for their dedication to improving global health and their willingness to share their data with the public.
https://github.com/MoH-Malaysia/covid19-public
