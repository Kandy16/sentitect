import os
import argparse

import pandas as pd
import re
from sklearn.utils import shuffle


# Removes stop words and other entities which are not likely meaningful words
def text_clean(text):

    for element in ["http\S+", "RT ", "[^a-zA-Z\'\.\,\d\s]", "[0-9]","\t", "\n", "\s+", "<.*?>"]:
        text = re.sub("r"+element, " ", text)

    return text


def main(args):
    train_df = pd.read_csv(os.path.join(args.data_path,"train-data.csv"))
    train_df = shuffle(train_df)[0:args.data_count]

    test_df = pd.read_csv(os.path.join(args.data_path,"test-data.csv"))
    test_df = shuffle(test_df)[0:args.data_count]

    # Create a merged data set and review initial information   
    combined_df = pd.concat([train_df, test_df])

     # Clean data sets
    combined_df.review = combined_df.review.apply(text_clean)

    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )
    parser.add_argument(
        '--data_count',
        type=int,
        help='Count of data to be processed. -1 refers to all'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path'
    )    
    args = parser.parse_args()
    if args.data_count == -1:
        args.data_count = None
    
    print('arguments:',args)
    result = main(args)
    print(result.head(5))
    result.to_csv(os.path.join(args.output,"output-data-prep.csv"), index=False)