import os
import argparse
import joblib

from azureml.core import Run

def run(args):
    
    print('--------Inside register file --------------------')
    print(args.data_path)
    print(os.listdir(args.data_path))

    runObj = Run.get_context()
    model = runObj.register_model(model_name='sentitect-best-model-pkl',
                                 model_path=os.path.join(args.data_path, 'sentitect-best-model-pkl.pkl'))
    return model

'sentitect-best-model-pkl.pkl'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path',
        dest='data_path',
        type=str,
        required=True,
        help='Path to the best model'
    )
    args = parser.parse_args()
    
    model = run(args)
    print(model.name, model.id, model.version, sep='\t')

    