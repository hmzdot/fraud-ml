# In this file we'll read data/fraud.csv for EDA
import pandas


def inspect_data():
    # Step 0. Read the data
    df = pandas.read_csv("data/fraud.csv")

    # Step 1. Print the first 5 rows
    print("First 5 rows of the data:")
    print(df.head())

    # Step 2. Print the shape of the data
    print("Shape of the data:")
    print(df.shape)

    # Step 3. Print the columns of the data
    print("Columns of the data:")
    print(df.columns)

    # Step 4. Print the data types of the columns
    print("Data types of the columns:")
    print(df.dtypes)

    # Step 5. Print the number of missing values in each column
    print("Number of missing values in each column:")
    print(df.isnull().sum())

    # Step 6. Print the summary statistics of the data, write to a file
    df.describe().to_csv("data/summary_statistics.csv")


if __name__ == "__main__":
    inspect_data()
