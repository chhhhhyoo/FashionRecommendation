
import pandas as pd

# Load the dataset
df = pd.read_csv('data/vali_modified.csv')

# Split the DataFrame based on the 'split' column
def split_and_save(df, path='data'):
    # Split the DataFrame into groups
    groups = df.groupby('split')

    # Save each group to a separate file
    for split_name, group in groups:
        filename = f'{path}/{split_name}_modified2.csv'
        group.to_csv(filename, index=False)
        print(f'Saved {split_name} split to {filename}')


# Split the DataFrame and save the splits to separate files
split_and_save(df)
