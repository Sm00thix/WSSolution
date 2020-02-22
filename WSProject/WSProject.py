from data_loader import *
from classifiers import *
import numpy as np
import random as rn
import tensorflow as tf

def perform_loading():
    print('Loading data...')
    questions, answers, category_ids, categories = load_data('WS/web_science_dataset.jsonl')
    print('Done!')
    unique_categories, unique_ids, unique_counts, mean_q_len, mean_a_len, std_q_len, std_a_len = stats(questions, answers, categories, category_ids)
    
    print('Unique categories:', unique_categories)
    print('Unique ids:', unique_ids)
    print('Counts:', unique_counts)
    print('Mean question length:', mean_q_len)
    print('Mean answer length:', mean_a_len)
    print('Standard deviation of question length:', std_q_len)
    print('Standard deviation of answer length:', std_a_len)

    return (questions, category_ids)

def perform_rf(x_train, y_train, x_val, y_val, x_test, y_test):
    print('Preprocessing dataset for random forest...')
    tokenizer = fit_tokenizer(x_train) # Fit the tokenizer on the training split
    x_train = get_bow(tokenizer, x_train) 
    x_val = get_bow(tokenizer, x_val)
    x_test = get_bow(tokenizer, x_test)
    print('Done!')

    print('Training a random forest classifier...')
    model, best_params = do_random_forest(x_train, y_train, x_val, y_val, x_test, y_test)
    print('Done!')
    print('Best parameters for random forest:', best_params)
    y_pred = model.predict(x_test)
    print('Evaluating best model:', eval(y_test, y_pred))

def perform_nn(x_train, y_train, x_val, y_val, x_test, y_test):
    print('Preprocessing dataset for LSTM NN...')
    x_train, x_val, x_test, emb_layer, dictionairy, vocab_length = get_embedding_weights(x_train, x_val, x_test)
    print('Done!')
    print('Training a neural classifier...')
    model = do_lstm(emb_layer, x_train, y_train, x_val, y_val, x_test, y_test)
    print('Done!')
    y_pred = model.predict_classes(x=x_test)
    #print('Evaluating best NN model:', eval(y_test, y_pred))
    return (eval(y_test, y_pred))

if __name__ == '__main__':
    #np.random.seed(27)
    #rn.seed(42)
    #tf.random.set_seed(69)
    #np.random.seed(221347)
    #rn.seed(421342)
    #tf.random.set_seed(691234)

    ################################
    #### Load & preprocess data ####
    ################################
    questions, category_ids = perform_loading()
    
    ###############################
    ####### Visualize data ########
    ###############################

    # TODO: Plot some statistics regarding the data

    ###############################
    #### Perform classification ###
    ###############################=
    print('Performing general preprocessing...')
    tokens = sanitize(questions)
    test_and_val_size=0.1
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(tokens, category_ids, test_and_val_size)
    print('Done!')
    perform_rf(x_train, y_train, x_val, y_val, x_test, y_test)
    perform_nn(x_train, y_train, x_val, y_val, x_test, y_test)