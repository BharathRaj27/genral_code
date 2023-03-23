import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats
from PIL import Image

def build_model(train_data, test_data):
    # Split the data into features and target
       X_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]
       X_test, y_test = test_data.iloc[:,:-1], test_data.iloc[:,-1]

       # Initialize the logistic regression model
       lr = LogisticRegression()

       # Train the model on the training data
       lr.fit(X_train, y_train)

       # Use the model to predict on the test data
       y_pred = lr.predict(X_test)

       # Calculate the accuracy of the model on the test data
       accuracy = accuracy_score(y_test, y_pred)   
       
       return accuracy

def label_encode(df, cols):
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df

def onehot_encode(df, cols):
    ohe = OneHotEncoder()
    for col in cols:
        encoded = pd.DataFrame(ohe.fit_transform(df[[col]]).toarray(), columns=[f'{col}_{i}' for i in range(ohe.categories_[0].shape[0])])
        df = pd.concat([df, encoded], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df

# Define function to handle missing values based on the selected technique
def handle_missing_values(df, technique):
    if technique == 'Drop':
        df.dropna(inplace=True)
    elif technique == 'Fill with mean':
        df.fillna(df.mean(), inplace=True)
    elif technique == 'Fill with median':
        df.fillna(df.median(), inplace=True)
    elif technique == 'Fill with mode':
        df.fillna(df.mode().iloc[0], inplace=True)
        
# Define a function to calculate the percentage of outliers
def calculate_outlier_percentage(column):
    q1, q3 = np.percentile(column, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers) / len(column) * 100

# Define a function to check for outliers in each numerical column
def check_outliers(df):
    numerical_columns = df.select_dtypes(include=np.number).columns
    outlier_percentages = {}
    for column in numerical_columns:
        outlier_percentages[column] = calculate_outlier_percentage(df[column])
    return outlier_percentages

# Function to remove outliers using z-score method
def remove_outliers_zscore(data):
    # Select a single column from the input data
    column = data.iloc[:, 0]
    
    # Convert column to numeric data type
    column_numeric = pd.to_numeric(column, errors='coerce')
    
    # Compute z-scores on numeric data
    z_scores = stats.zscore(column_numeric)
    
    # Identify outliers
    outliers = np.abs(z_scores) > 3
    
    # Remove outliers from original data
    column[~outliers] = column_numeric[~outliers]
    
    return data

# Function to remove outliers using IQR method
def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return filtered_data


# Title of the app
st.title('Logistic Regression:')
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStcD2Yl6mWPOd0tpEHexjKvYYZFTb9-ow5Ug&usqp=CAU")

import warnings

# Ignore warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings('ignore')

def main():
  # Set up a file uploader with multiple file selection enabled
  uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success('File successfully uploaded!')

    # Display raw data
    if st.checkbox('Display Raw Data'):
        st.write(df.head())
    
    # Data Types
    if st.checkbox('Data Information'):
        # Display dataset metadata
        st.write('## Dataset Info')
        st.write(f"Number of Rows: {df.shape[0]}")
        st.write(f"Number of Columns: {df.shape[1]}")
        st.write('---')
        st.write('### Data Types:')
        st.write(df.dtypes)
        st.write('---')
        st.write('### Descriptive Statistics:')
        st.write(df.describe())
        # Identify numerical and categorical columns
        num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
        st.write('### Numerical Columns:', num_cols)
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        st.write('### Categorical Columns:', cat_cols)
        

    st.title('Data Preprocessing')
    st.subheader('Drop Unwanted Columns')
    # Allow users to select columns to drop
    if st.checkbox('drop'):
        st.write("###### Drop Columns which are not Required")
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        df = df.drop(columns_to_drop, axis=1)
        st.success("Columns dropped successfully!")
    
        if st.checkbox('Show dataset'):
            # Display the filtered dataframe
            st.dataframe(df)
        
    # Create the Streamlit app
    st.subheader('Missing Values Handling')   
    # Create a checkbox to check for missing values
    if st.checkbox('Check missing values'):
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.write(missing_values)
            st.warning('Missing Values Present')
        else:
            st.success('No Missing Values!')
         
    # Create a checkbox to handle missing values
    if st.checkbox('Handle Missing Values'):
        # Create a selectbox to choose the technique
        technique = st.selectbox('Choose a technique', ('Drop', 'Fill with mean', 'Fill with median', 'Fill with mode'))
        # Show the original dataframe
        st.subheader('Original Dataframe')
        st.write(df)
        # Apply the selected technique to the dataframe
        handle_missing_values(df, technique)
        # Show the modified dataframe
        st.subheader(f'Dataframe after {technique} technique')
        st.write(df)

    # Check for outliers
    st.subheader('Checking & Handling Outliers')
    if st.checkbox('Check for outliers'):
        outlier_percentages = check_outliers(df)
        for column, percentage in outlier_percentages.items():
            st.write(f"{column}: {percentage:.2f}% outliers")
        
        # Create a checkbox to show/hide outliers
        show_outliers = st.checkbox('Show outliers')
        if show_outliers:
            # Find the column with the highest outliers
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            highest_outliers_col = None
            highest_outliers_percent = 0
    
            for column in numerical_cols:
                outlier_percent = calculate_outlier_percentage(df[column])
                if outlier_percent > highest_outliers_percent:
                    highest_outliers_col = column
                    highest_outliers_percent = outlier_percent
            
            # Create a box plot of the highest outlier column
            fig, ax = plt.subplots()
            ax.boxplot(df[highest_outliers_col])
            ax.set_title('Box plot of ' + highest_outliers_col)
            ax.set_ylabel(highest_outliers_col)
            st.pyplot(fig)
       
    # Create a checkbox to remove outliers using z-score method
    if st.checkbox('Remove Outliers'):
        remove_outliers_zscore_checkbox = st.checkbox('z-score method')
        if remove_outliers_zscore_checkbox:
            filtered_df = remove_outliers_zscore(df)
            st.write('Data after removing outliers using z-score method:')
            st.write(filtered_df)

        # Create a checkbox to remove outliers using IQR method
        remove_outliers_iqr_checkbox = st.checkbox('IQR method')
        if remove_outliers_iqr_checkbox:
            filtered_df = remove_outliers_iqr(df)
            st.write('Data after removing outliers using IQR method:')
            st.write(filtered_df)

    st.subheader('Converting Categorical to Numerical')
    cols = st.multiselect("Select columns to encode", df.columns.tolist())
    # Select encoding method
    encode_method = st.radio("Select encoding method", ["Label Encoding", "One-Hot Encoding"])
    # Encode columns
    if cols:
        if encode_method == "Label Encoding":
            df = label_encode(df, cols)
            print('Label Encoded',df)
        elif encode_method == "One-Hot Encoding":
            df = onehot_encode(df, cols)
            print('One-hot encoded data', df)

    # Display encoded data
    st.write('##### Displaying Encoded Data',df)

    st.title('Data Visualization')

    Plots = ["Histogram", "Correlation Matrix", "Scatter plot","Bar plot"]
    selected_Plot = st.selectbox("Choose a Plot for Visualization", Plots)


    if selected_Plot == "Histogram":
        selected_column = st.selectbox('Select Column', df.columns)
        plt.hist(df[selected_column])
        st.pyplot()

    elif selected_Plot == "Correlation Matrix":
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True)
        st.pyplot()

    elif selected_Plot == "Scatter plot":
        x = st.selectbox('Select the x-axis for scatter plot', df.columns)
        y = st.selectbox('Select the y-axis for scatter plot', df.columns)
        plt.scatter(df[x], df[y])
        st.pyplot()

    elif selected_Plot == "Bar plot":
        col = st.selectbox('Select the column for bar chart', df.columns)
        plt.bar(df[col].value_counts().index, df[col].value_counts())
        st.pyplot()
    
    st.title('Feature Engineering')
    # Create selectbox for target variable
    target_variable = st.selectbox("Select target variable", df.columns)

    # Use target variable in model building
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    # Split data
    st.subheader('Data spltting')
    st.write("#### Spliting The Data")
    test_size = st.slider("Test set size", 0.0, 1.0, 0.2, 0.1)
    random_state = st.slider("Random state", 0, 100, 42, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Display encoded data
    if st.checkbox('Training data'):
           st.write("Training data:")
           st.write('X Train',X_train)
           st.write('Y Train',y_train)



    if st.checkbox('Testing data'):
        st.write("Testing data:")
        st.write("X Test",X_test)
        st.write("Y Test",y_test)

    st.title('Model building')
    if st.checkbox('Accuracy'):
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        accuracy = build_model(train_data, test_data)
        st.write(f"Accuracy: {accuracy}")




# Run the Streamlit app
if __name__ == '__main__':
    main()
