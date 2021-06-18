import data_prep as dprep
import vectorize as vect

class something:
    pass
args = something()
args.data_path ='../data'
args.data_count= 10

prepared_data = dprep.main(args)
vectorized_data = vect.main(prepared_data)


print(vectorized_data)