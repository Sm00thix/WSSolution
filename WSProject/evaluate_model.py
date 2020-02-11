from tensorflow import keras
from classifiers import eval

def eval_model(model_path, x_test, y_test):
    print("Loading model...")
    model = keras.models.load_model(model_path)
    y_pred = model.predict_classes(x_test)
    score = eval(y_test, y_pred)
    return score
