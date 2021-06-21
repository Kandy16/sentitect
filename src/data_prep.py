import os
import argparse

import pandas as pd
import re
from sklearn.utils import shuffle

from azureml.core import Run


# Removes stop words and other entities which are not likely meaningful words
def text_clean(text):

    for element in ["http\S+", "RT ", "[^a-zA-Z\'\.\,\d\s]", "[0-9]","\t", "\n", "\s+", "<.*?>"]:
        text = re.sub("r"+element, " ", text)

    return text


def run(args):
    df_list = []
    print(f'Data path: {args.data_path}')

    #data_path = Run.get_context().input_datasets['input_data']

    #print(f'Listing files inside directory {data_path}')
    for file1 in os.listdir(args.data_path):
        print(file1)

    for file1 in os.listdir(args.data_path):
        if(file1.endswith('.csv')):
            df = pd.read_csv(os.path.join(args.data_path,file1))
            df = shuffle(df)[0:args.data_count]
            df_list.append(df)

    # Create a merged data set and review initial information   
    combined_df = pd.concat(df_list)

     # Clean data sets
    combined_df.review = combined_df.review.apply(text_clean)

    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path',
        dest='data_path',
        type=str,
        required=True,
        help='Path to the training data'
    )
    parser.add_argument(
        '--data-count',
        dest='data_count',
        type=int,
        nargs='?',
        const=-1,
        help='Count of data to be processed. -1 refers to all'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path'
    )    
    args = parser.parse_args()
    if args.data_count == -1:
        args.data_count = None
    
    print('arguments:',args)
    result = run(args)
    print(result.head(5))
    result.to_csv(os.path.join(args.output,"output-data-prep.csv"), index=False)