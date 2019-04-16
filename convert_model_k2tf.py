import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', default='model')
    parser.add_argument('--resume', '-r')
    args = parser.parse_args()

    if args.resume is not None:
        model = load_model(args.resume)
        tf.contrib.saved_model.save_keras_model(model, args.model)
