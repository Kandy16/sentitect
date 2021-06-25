import os
import argparse


def run(args):
    count = 0

    for file1 in os.listdir(args.data_path):
        print(file1)
        count += 1

    return count

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
        '--output',
        type=str,
        required=True,
        help='Output path'
    )    
    args = parser.parse_args()
 
    
    print('arguments:',args)
    result = run(args)

    print('The answer is : ', result)
    with open(os.path.join(args.output, 'count-files.txt'),'w') as f:
        f.write(str(result))

    
