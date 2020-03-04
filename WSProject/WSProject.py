from data_loader import *
from classifiers import *
from sklearn.metrics import mean_squared_error
import gensim
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

def perform_nn(x_train, y_train, x_val, y_val, x_test, y_test, gensim_model):
    print('Preprocessing dataset for LSTM NN...')
    x_train, x_val, x_test, emb_layer, dictionairy, vocab_length = get_embedding_weights(x_train, x_val, x_test, gensim_model)
    max_x = np.amax(x_train)
    print('Done!')

    #preds = np.empty((10,7))
    #for i in range(10):
    #    print('Training a neural classifier...')
    #    model = do_lstm(emb_layer, x_train, y_train, x_val, y_val, max_x, 5)
    #    print('Done!')
    #    y_pred = model.predict_classes(x=x_test)
    #    preds[i] = eval(y_test, y_pred)
    #print('Average metrics of 10 best NN models:', np.mean(preds, axis=1))
    #return

    print('Training a neural classifier...')
    model = do_lstm(emb_layer, x_train, y_train, x_val, y_val, max_x, 5)
    print('Done!')
    y_pred = model.predict_classes(x=x_test)
    return(eval(y_test, y_pred))

def load_crowdsource():
    print('Loading data from dataset provided by TAs...')
    questions, answers, category_ids, categories, question_ids, answer_labels, answer_qualities = load_data('WS/web_science_dataset_with_labels.jsonl', True)
    print('Done!')
    return(question_ids, answers, answer_qualities)

def clean_crowdsource():
    print('Loading crowdsourced dataset')
    df = load_cs_csvs('WS/submissions_fixed_anonymized')
    print('Done!')
    clean_and_stats_dataframe(df)
    return

def perform_cs_nn(question_ids, answers, answer_qualities, gensim_model, first_or_last_tokens):
    v_int = np.vectorize(int)
    train_ids = v_int(np.loadtxt('WS/training_ids.txt'))
    test_ids = v_int(np.loadtxt('WS/testing_ids.txt'))
    tokens = sanitize(answers)
    question_ids = question_ids.tolist()
    train_idxs = [question_ids.index(train_id) for train_id in train_ids]
    test_idxs = [question_ids.index(test_id) for test_id in test_ids]
    original_x_train = tokens[train_idxs]
    x_test = tokens[test_idxs]
    original_y_train = answer_qualities[train_idxs] - 1
    y_test = answer_qualities[test_idxs] - 1
    val_size = 0.1

    preds = np.empty((10,7))
    for i in range(10):
        x_train, x_val, y_train, y_val = train_test_split(original_x_train, original_y_train, stratify=original_y_train, test_size=val_size)
        x_train, x_val, x_test, emb_layer, dictionairy, vocab_length = get_embedding_weights(x_train, x_val, x_test, gensim_model, first_or_last_tokens)
        max_x = np.amax(x_train)
        print('Training a neural classifier...')
        model = do_lstm(emb_layer, x_train, y_train, x_val, y_val, max_x, 3)
        print('Done!')
        y_pred = model.predict_classes(x=x_test)
        preds[i] = eval(y_test, y_pred)
    print('Average metrics of 10 best NN models:', np.mean(preds, axis=1))
    print('Std. of metrics of 10 best NN models:', np.std(preds, axis=1))

    reg_mses = np.empty((10))
    for j in range(10):
        print('Training a neural regressor...')
        x_train, x_val, y_train, y_val = train_test_split(original_x_train, original_y_train, stratify=original_y_train, test_size=val_size)
        x_train, x_val, x_test, emb_layer, dictionairy, vocab_length = get_embedding_weights(x_train, x_val, x_test, gensim_model, first_or_last_tokens)
        max_x = np.amax(x_train)
        reg_model = do_lstm_regress(emb_layer, x_train, y_train, x_val, y_val, max_x)
        print('Done!')
        reg_y_pred = reg_model.predict(x=x_test)
        reg_mses[j] = mean_squared_error(reg_y_pred, y_test)
    print('Average mean squared error of 10 best NN regressors:', np.mean(reg_mses))
    print('Std. of metrics of 10 best NN regressors:', np.std(reg_mses))
    return

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
    print("Loading word embeddings...")
    filename = 'WS/GoogleNews-vectors-negative300'
    try:
        gensim_model = gensim.models.KeyedVectors.load(filename, mmap='r')
    except:
        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(filename + '.bin', binary=True)
        gensim_model.wv.save(filename)

    print("Done loading word embeddings!")
    questions, category_ids = perform_loading()

    ###############################
    #### Perform classification ###
    ###############################
    nn_experiments = 10
    nn_metrics = 7
    test_and_val_size=0.1

    print('Performing general preprocessing...')
    tokens = sanitize(questions)
    print('Done!')
    preds = np.empty((nn_experiments,nn_metrics))
    for i in range(nn_experiments):
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(tokens, category_ids, test_and_val_size)
        preds[i] = perform_nn(x_train, y_train, x_val, y_val, x_test, y_test, gensim_model)
    print('Average metrics of 10 best NN models:', np.mean(preds, axis=1))
    print('Std. of metrics of 10 best NN models:', np.std(preds, axis=1))

    perform_rf(x_train, y_train, x_val, y_val, x_test, y_test)
    ##############################
    ######## Crowdsourcing #######
    ##############################
    clean_crowdsource()
    question_ids, answers, answer_qualities = load_crowdsource()
    perform_cs_nn(question_ids, answers, answer_qualities, gensim_model, 'first')
    perform_cs_nn(question_ids, answers, answer_qualities, gensim_model, 'last')
