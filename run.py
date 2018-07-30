"""juno

Usage:
    run.py [--db] (--sd04|--sd08) [-a <float>] [-e <int>] [-T]

Options:
    -h --help                          help
    --db                               build database
    --sd04                             use images from sd04
    --sd08                             use images from sd08
    -a <float> --augmentation <float>  probability of augmentation for each image in training [defaultL .5]
    -e <int> --epochs <int>            number of epochs [default: 16]
    -T                                 test run


"""
from docopt import docopt
from utils import set_up_db
import classification_keras


def parse_env_var(env_var):
    if env_var['--sd04']:
        env_var['image_set'] = 'sd04'
    if env_var['--sd08']:
        env_var['image_set'] = 'sd08'
    return env_var

def main(image_set, db_build, augmentation, epochs, test):
    pass


if __name__ == '__main__':
    env_var = parse_env_var(docopt(__doc__))
    print(env_var)
    # set_up_db.read_files_sd04()
    # main(image_set=env_var[''])
    vv = set_up_db.SD04()
    vv.create_csv()
    vv.csv_to_dict()
    vv.enhance_images_to_rgb()
    # set_up_db.read_files_sd04()
    # set_up_db.read_txt_to_dict()
    ids_cat, mapping = set_up_db.read_txt_to_dict()
    neural_net = classification_keras.train_neural_net(ids_cat=ids_cat, mapping=mapping)
    classification_keras.predict_neural_net(model=neural_net, ids_cat=ids_cat, mapping=mapping)

