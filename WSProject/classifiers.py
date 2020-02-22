from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from data_loader import get_embedding_weights
import numpy as np
from tensorflow import keras
from sklearn.utils import class_weight

def eval(y_true, y_pred):
    """
    Calculates different scoring metrics for a prediction.\n
    y_true: the true labels\n
    y_pred: the predicted labels\n
    """
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_weigthed = recall_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return ([precision_macro, precision_weighted, recall_macro, recall_weigthed, f1_macro, f1_weighted, accuracy])

def split_dataset(x_set, y_set, test_and_val_size):
    """
    Splits the dataset in to training, validation and test samples. Validation and test samples each consist of a fraction of the enite dataset equal to test_and_val_size.
    """
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, stratify=y_set, test_size=test_and_val_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=(len(x_test) / len(x_train)))
    return (x_train, y_train, x_val, y_val, x_test, y_test)

def do_random_forest(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Performs validation with a grid search of 100 different parameters, resulting in 100 different models.\n
    x_train: the training data samples\n
    y_train: the training data labels\n
    x_val: the validation data samples\n
    y_val: the validation data labels\n
    x_test: the test data samples\n
    y_test: the test data labels\n
    """
    clf = RandomForestClassifier()
    no_features = x_train.shape[1]
    param_grid = {
            'n_estimators': [50 * i for i in range(1,11)],
            'max_features': [int(round(i * no_features / 5)) for i in range (1,6)],
            'oob_score': [True],
            'class_weight': ['balanced', None]
        }
    largest_train_index = len(x_train)
    train_indices = np.array([-1 for i in range(largest_train_index)])
    x_train = np.concatenate((x_train, x_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    val_indices = np.array([0 for i in range(largest_train_index, len(x_train))])
    fold_indices = np.concatenate((train_indices, val_indices), axis=0)
    ps = PredefinedSplit(fold_indices)
    gs = GridSearchCV(clf, param_grid, cv=ps, scoring='f1_weighted', n_jobs=-1, verbose=100)
    gs.fit(x_train, y_train)
    best_params = gs.best_params_
    return (gs, best_params)

def do_lstm(emb_layer, x_train, y_train, x_val, y_val, x_test, y_test):
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)
    #checkpoint = keras.callbacks.ModelCheckpoint('model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)
    callbacks_list = [earlystopping]
    model = keras.models.Sequential()
    embedding_layer = keras.layers.Embedding(np.amax(x_train) + 1,
            emb_layer.shape[1],
            weights=[emb_layer],
            input_length=x_train.shape[1],
            trainable=False)
    model.add(embedding_layer)
    model.add(keras.layers.LSTM(256))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=200, validation_data=(x_val, y_val), callbacks=callbacks_list, class_weight=class_weights, batch_size=len(x_train))
    return (model)