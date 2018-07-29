"""juno

Usage:
    run.py [-b] (--sd04|--sd08) [-e <int>] [-T]

Options:
    -h --help                help
    -b                       build database
    --sd04                   use images from sd04
    --sd08                   use images from sd08
    -e <int> --epochs <int>  number of epochs [default: 16]
    -T                       test run


"""
from docopt import docopt
from utils import set_up_db
import classification_keras

if __name__ == '__main__':
    env_var = docopt(__doc__)
    print(env_var)
    # set_up_db.read_files_sd04()
    quit(-9)
    vv = set_up_db.SD04()
    vv.create_csv()
    vv.csv_to_dict()
    vv.enhance_images_to_rgb()
    # set_up_db.read_files_sd04()
    # ids_cat, mapping = set_up_db.read_csv_to_dict()
    neural_net = classification_keras.train_neural_net(ids_cat=vv.sample_y, mapping=vv.mapping)
    # classification_keras.predict_neural_net(model=neural_net, ids_cat=ids_cat, mapping=mapping)

