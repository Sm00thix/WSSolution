from data_loader import *
from classifiers import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import pandas as pd
import numpy as np
import random as rn
import tensorflow as tf
import multiprocessing as mp
import itertools

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

    print('Training a neural classifier...')
    model = do_lstm(emb_layer, x_train, y_train, x_val, y_val, max_x, 5)
    print('Done!')
    y_pred = model.predict_classes(x=x_test)
    return(eval(y_test, y_pred))

def load_crowdsource():
    print('Loading data from dataset provided by TAs...')
    questions, answers, category_ids, categories, question_ids, answer_labels, answer_qualities = load_data('WS/web_science_dataset_with_labels.jsonl', True)
    print('Done!')
    return(questions, question_ids, categories, answers, answer_qualities)

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
    print(f'Using {first_or_last_tokens} tokens')
    print('Average metrics of 10 best NN models:', np.mean(preds, axis=0))
    print('Std. of metrics of 10 best NN models:', np.std(preds, axis=0))

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
    print(f'Using {first_or_last_tokens} tokens')
    print('Average mean squared error of 10 best NN regressors:', np.mean(reg_mses))
    print('Std. of metrics of 10 best NN regressors:', np.std(reg_mses))
    return

def perform_recommend(questions, question_ids, categories, train_df, test_df, k, top_n_words):

    def aggregate_recommends(recommends, not_recommends):
        # Compute the set of questions to recommend. This is all questions that have been recommended more than they have been not_recommended
        f_rec, rec_counts = np.unique(np.concatenate(recommends), return_counts=True)
        f_not_rec, not_rec_counts = np.unique(np.concatenate(not_recommends), return_counts=True)
        f_not_rec = f_not_rec.tolist()
        n_recs = []
        for item in f_rec:
            try:
                val = not_rec_counts[f_not_rec.index(item)]
                n_recs.append(val)
            except:
                n_recs.append(0)
        final = [f_rec[i] for i in range(f_rec.shape[0]) if rec_counts[i] > n_recs[i]]
        return final

    def compute_accuracy(recommendations, test_recommends, test_not_recommends):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(recommendations)):
            cur_tp = TP # True positives up until this user
            cur_fp = FP # False positives up until this user
            for rec in recommendations[i]:
                if rec in test_recommends[i]:
                    TP += 1
                elif rec in test_not_recommends[i]:
                    FP += 1
            FN += len(test_recommends[i]) - TP + cur_tp # Update false negatives with false negatives for current user
            TN += len(test_not_recommends[i]) - FP + cur_fp # Update true negatives with false negatives for current user
        acc = (TP + TN) / (TP + TN + FP + FN)
        return acc

    # compute tfidf and document vectors
    tokens = sanitize(questions)
    tfidf, vocab = get_tfidf_vocab(tokens, top_n_words)
    
    #top_n_tfidf_idxs = np.take(np.flip(np.argsort(tfidf, axis=1), axis=1), range(top_n_words), axis=1)
    #keys = list(vocab.keys())
    #values = list(vocab.values())
    #reverse_dict = dict(zip(values, keys))
    vector_len = tfidf.shape[1]

    #top_n_vectors = np.empty((top_n_tfidf_idxs.shape[0], top_n_tfidf_idxs.shape[1], vector_len))
    #broken_words_idxs = []
    #for i in range(top_n_tfidf_idxs.shape[0]):
    #    for j in range(top_n_tfidf_idxs.shape[1]):
    #        idx = top_n_tfidf_idxs[i,j]
    #        word = reverse_dict[idx]
    #        try:
    #            top_n_vectors[i,j] = gensim_model.word_vec(word)
    #        except:
    #            broken_words_idxs.append((i,j))
    #average_vectors = np.zeros((top_n_tfidf_idxs.shape[0], vector_len))
    #for i in range(top_n_tfidf_idxs.shape[0]):
    #    no_actual_vectors = 0
    #    for j in range(top_n_tfidf_idxs.shape[1]):
    #        if (i,j) not in broken_words_idxs and j < len(tokens[i]):
    #            average_vectors[i] += top_n_vectors[i,j]
    #            no_actual_vectors += 1
    #        else:
    #            continue
    #    average_vectors[i] /= no_actual_vectors
                

    #compute average word embedding for each user based on top_n_tfidf_idxs
    #only consider word embedding for min(top_n_words, len(tokens[i]) for each document - otherwise we introduce noise
    # use average vectors instead of tfidf
    users = pd.unique(train_df['userID'])

    def compute_scores(train_category, test_category, k, top_n_words):
        train_question_ids = question_ids[categories == train_category]
        test_question_ids = question_ids[categories == test_category]
        # must split query in two to avoid scoping issues
        sub_train_df = train_df.query('questionID in @train_question_ids')
        users_likes = [sub_train_df.query('userID == @user and rating == 3')['questionID'] for user in users]

        sub_recommend_df = train_df.query('questionID in @test_question_ids')
        users_recommends = np.array([sub_recommend_df.query('userID == @user and rating == 3')['questionID'].to_numpy() for user in users])
        users_not_recommends = np.array([sub_recommend_df.query('userID == @user and rating == 1')['questionID'].to_numpy() for user in users])

        sub_test_df = test_df.query('questionID in @test_question_ids')
        sub_test_recommend_df = [sub_test_df.query('userID == @user and recommend == "Yes"')['questionID'].to_numpy() for user in users]
        sub_test_not_recommend_df = [sub_test_df.query('userID == @user and recommend == "No"')['questionID'].to_numpy() for user in users]

        users_vectors = []
        for i in range(len(users)):
            doc_reps = [tfidf[np.where(question_ids==id)[0][0]] for id in users_likes[i]]
            if len(doc_reps) == 0:
                users_vectors.append(np.zeros(vector_len))
            else:
                users_vectors.append(np.average(doc_reps, axis=0))

        cos_sims = cosine_similarity(np.array(users_vectors))
        top_k_friends = np.empty((len(users), k)) # assumption: 0 < k < len(users)
        recommendations = [[] for _ in range(len(users))]
        for i in range(len(users)):
            #if len(users_likes[i]) == 0:
            #    # If the user does not like anything within the current training topic, we do not recommend anything because we can not compute nearest friends correctly.
            #    recommendations[i].extend([])
            #else:
            friends = users[np.argsort(cos_sims[i])[::-1][1:k+1]]
            top_k_friends[i] = friends
            friends_idxs = [np.where(users==friend)[0][0] for friend in friends]
            friends_recommends = users_recommends[friends_idxs]
            friends_not_recommends = users_not_recommends[friends_idxs]
            recommendations[i].extend(aggregate_recommends(friends_recommends, friends_not_recommends))
        # recommendations have now been computed.
        # compute the accuracy between recommendations and actual recommended items (TP + TN) / (TP + TN + FP + FN), TN is any "no" in the test set that was not recommended.
        # Save result in the result matrix
        acc = compute_accuracy(recommendations, sub_test_recommend_df, sub_test_not_recommend_df)
        return acc
    uniq_cats = np.unique(categories)
    accuracy_matrix = np.empty((len(uniq_cats), len(uniq_cats)))
    for i in range(len(uniq_cats)):
        for j in range(len(uniq_cats)):
            accuracy_matrix[i][j] = compute_scores(uniq_cats[i], uniq_cats[j], k, top_n_words)
            #print('Done for train: ', uniq_cats[i], ' and test: ', uniq_cats[j], ' result = ', accuracy_matrix[i][j])
    return accuracy_matrix

def caller(params):
    questions, question_ids, categories, train_df, test_df, k_friends, top_n_words = params
    print('Computing with k: ', k_friends, ' and n: ', top_n_words)
    acc_matrix = perform_recommend(questions, question_ids, categories, train_df, test_df, k_friends, top_n_words)
    avg = np.average(acc_matrix)
    return (avg, k_friends, top_n_words)

def perform_recommend_search(questions, question_ids, categories):
    k_friends = range(1, 29)
    top_n_words = range(100, 3200, 50)
    avg_acc_matrix = np.empty((len(k_friends), len(top_n_words)))
    p = mp.Pool(12)
    train_df, test_df = load_tsvs('WS/')
    params = list(itertools.product(k_friends, top_n_words))
    params = [(questions, question_ids, categories, train_df, test_df, k_friends, top_n_words) for (k_friends, top_n_words) in params]
    result = p.map(caller, params)
    p.close()
    p.join()
    best_acc = 0
    best_k = 0
    best_n = 0
    for vals in result:
        avg, k_friends, top_n_words = vals
        if avg > best_acc:
            best_acc = avg
            best_k = k_friends
            best_n = top_n_words
    print('Final best parameters:')
    print('Best k = ', best_k)
    print('Best n = ', best_n)
    print('Best average accuracy: ', best_acc)
    print('Train by test categories are: ', np.unique(categories))
    best_acc_matrix = perform_recommend(questions, question_ids, categories, train_df, test_df, best_k, best_n)
    print('Best accuracy matrix: ', best_acc_matrix)

    #best_acc = 0
    #best_k = 0
    #best_n = 0
    #        acc_matrix = perform_recommend(questions, question_ids, categories, train_df, test_df, k_friends, top_n_words)
    #        avg = np.average(acc_matrix)
    #        if avg > best_acc:
    #            best_acc_matrix = acc_matrix
    #            best_acc = avg
    #            best_k = k_friends
    #            best_n = top_n_words
    #            print('Found best_k: ', k_friends, ' and top_n_words: ', top_n_words, ' with acc: ', avg)
    #print('Final best parameters:')
    #print('Best k = ', best_k)
    #print('Best n = ', best_n)
    #print('Best average accuracy: ', best_acc)
    #print('Train by test categories are: ', np.unique(categories))
    #print('Best accuracy matrix: ', best_acc_matrix)

if __name__ == '__main__':
    ################################
    #### Load & preprocess data ####
    ################################
    #print("Loading word embeddings...")
    #filename = 'WS/GoogleNews-vectors-negative300'
    #try:
    #    gensim_model = gensim.models.KeyedVectors.load(filename, mmap='r')
    #except:
    #    gensim_model = gensim.models.KeyedVectors.load_word2vec_format(filename + '.bin', binary=True)
    #    gensim_model.wv.save(filename)

    #print("Done loading word embeddings!")
    #questions, category_ids = perform_loading()

    ###############################
    #### Perform classification ###
    ###############################
    #nn_experiments = 10
    #nn_metrics = 7
    #test_and_val_size = 0.1

    #print('Performing general preprocessing...')
    #tokens = sanitize(questions)
    #print('Done!')
    #preds = np.empty((nn_experiments,nn_metrics))
    #for i in range(nn_experiments):
    #    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(tokens, category_ids, test_and_val_size)
    #    preds[i] = perform_nn(x_train, y_train, x_val, y_val, x_test, y_test, gensim_model)
    #print('Average metrics of 10 best NN models:', np.mean(preds, axis=0))
    #print('Std. of metrics of 10 best NN models:', np.std(preds, axis=0))

    #perform_rf(x_train, y_train, x_val, y_val, x_test, y_test)

    ##############################
    ######## Crowdsourcing #######
    ##############################
    #clean_crowdsource()
    questions, question_ids, categories, answers, answer_qualities = load_crowdsource()
    #perform_cs_nn(question_ids, answers, answer_qualities, gensim_model, 'first')
    #perform_cs_nn(question_ids, answers, answer_qualities, gensim_model, 'last')

    ##############################
    ######## Recommender #########
    ##############################

    perform_recommend_search(questions, question_ids, categories)