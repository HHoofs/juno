from utils import set_up_db
import classification_keras

if __name__ == '__main__':
    # set_up_db.read_files_sd04()
    ids_cat, mapping = set_up_db.read_csv_to_dict()
    neural_net = classification_keras.train_neural_net(ids_cat=ids_cat, mapping=mapping)
    # classification_keras.predict_neural_net(model=neural_net, ids_cat=ids_cat, mapping=mapping)
