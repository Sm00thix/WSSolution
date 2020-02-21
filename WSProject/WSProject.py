from data_loader import *
from classifiers import *
from evaluate_model import *
import numpy as np
import random as rn
import tensorflow as tf

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
    print('Loading data...')
    questions, answers, category_ids, categories = load_data('WS/web_science_dataset.jsonl')
    unique_categories, unique_ids, unique_counts, mean_q_len, mean_a_len, std_q_len, std_a_len = stats(questions, answers, categories, category_ids)
    
    print('Unique categories:', unique_categories)
    print('Unique ids:', unique_ids)
    print('Counts:', unique_counts)
    print('Mean question length:', mean_q_len)
    print('Mean answer length:', mean_a_len)
    print('Standard deviation of question length:', std_q_len)
    print('Standard deviation of answer length:', std_a_len)
    
    ###############################
    #### Perform classification ###
    ###############################
    print('Preprocessing dataset for random forest...')
    tokens = sanitize(questions)
    test_size=0.2
    rf_x_train, rf_y_train, rf_x_test, rf_y_test = split_dataset(tokens, category_ids, test_size)
    tokenizer = fit_tokenizer(rf_x_train) # Fit the tokenizer on the training split
    rf_x_train = get_bow(tokenizer, rf_x_train) 
    rf_x_test = get_bow(tokenizer, rf_x_test)

    
    #print('Performing random forest classification...')
    #rf_pred, best_params, train_score = do_random_forest(rf_x_train, rf_y_train, rf_x_test, rf_y_test)
    #print('Best parameters for random forest:', best_params)
    #print('Training f1 weigthed score for random forest', train_score)
    #print('Evaluating best model:', eval(rf_y_test, rf_pred))
    

    print('Preprocessing dataset for LSTM NN...')
    test_size=0.1
    x_train, y_train, x_test, y_test = split_dataset(tokens, category_ids, test_size)
    nn_x_train, nn_x_test, nn_emb_layer, dictionairy, vocab_length = get_embedding_weights(x_train, x_test)
    nn_pred = do_lstm(nn_emb_layer, nn_x_train, y_train, nn_x_test, y_test)
    print('Evaluating best NN model:', eval(y_test, nn_pred))
    #nn_x_eval = get_vecs(tokens, dictionairy, vocab_length)
    #res = eval_model('model.hdf5', nn_x_eval, category_ids)
    #print(res)