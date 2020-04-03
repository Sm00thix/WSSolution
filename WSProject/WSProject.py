from data_loader import *
from classifiers import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import pandas as pd
import numpy as np
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
    model, best_params = do_random_forest(x_train, y_train, x_val, y_val)
    print('Done!')
    print('Best parameters for random forest:', best_params)
    y_pred = model.predict(x_test)
    print('Evaluating best model:', eval(y_test, y_pred))
    return (model, y_pred)

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
    return(questions, question_ids, categories, answers, answer_qualities, answer_labels)

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
            TN += len(test_not_recommends[i]) - FP + cur_fp # Update true negatives with true negatives for current user
        acc = (TP + TN) / (TP + TN + FP + FN)
        return acc

    # compute tfidf and document vectors
    tokens = sanitize(questions)
    tfidf = get_tfidf(tokens, top_n_words)
    vector_len = tfidf.shape[1]
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
    p = mp.Pool(mp.cpu_count()) # Use all virtual cpu cores
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

def perform_answer_detection(question_ids, answers, answer_labels, categories, gensim_model, top_tokens):

    # Computes a numerical representation of answer labels. yes -> 0; no -> 1, na -> 2
    def label_to_int(labels):
        def mapping(x):
            if x == 'yes':
                return 0
            elif x == 'no':
                return 1
            elif x == 'na':
                return 2
            else:
                raise Exception(f'Can not map label {x} to int')
        
        int_labels = np.vectorize(mapping)(labels)
        return int_labels
    v_int = np.vectorize(int)
    train_ids = v_int(np.loadtxt('WS/training_ids.txt'))
    test_ids = v_int(np.loadtxt('WS/testing_ids.txt'))
    tokens = sanitize(answers)
    question_ids = question_ids.tolist()
    train_idxs = [question_ids.index(train_id) for train_id in train_ids]
    test_idxs = [question_ids.index(test_id) for test_id in test_ids]
    train_cats = np.array([categories[idx] for idx in train_idxs])
    test_cats = np.array([categories[idx] for idx in test_idxs])
    original_x_train = tokens[train_idxs]
    x_test = tokens[test_idxs]
    original_y_train = label_to_int(answer_labels[train_idxs])
    y_test = label_to_int(answer_labels[test_idxs])
    val_size = 0.1

    def train_eval_sub_topics(x_train, y_train, x_test, y_test, val_size, gensim_model, top_tokens):
        
        def train_eval(sub_x_train, sub_y_train, category_x_test, category_y_test, uniq_vals, train_cat):
            sub_x_train, sub_x_val, sub_y_train, sub_y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=val_size)
            t = fit_tokenizer(sub_x_train)
            print('Training a random forest classifier...')
            rf_model, _ = perform_rf(sub_x_train, sub_y_train, sub_x_val, sub_y_val, sub_x_train, sub_y_train)
            print('Done!')

            nn_sub_x_train, nn_sub_x_val, nn_sub_x_test, emb_layer, dictionairy, vocab_length = get_embedding_weights(sub_x_train, sub_x_val, sub_x_train, gensim_model, 'last')
            max_x = np.amax(nn_sub_x_train)
            print('Training a neural classifier...')
            nn_model = do_lstm(emb_layer, nn_sub_x_train, sub_y_train, nn_sub_x_val, sub_y_val, max_x, 3)
            print('Done!')

            nn_results = np.empty(len(uniq_vals))
            rf_results = np.empty(len(uniq_vals))
            for i in range(len(uniq_cats)):
                test_cat = uniq_cats[i]
                sub_x_test = category_x_test[i]
                sub_y_test = category_y_test[i]
                np.savetxt(f'{test_cat}_true', sub_y_test)

                rf_result, rf_y_pred = eval_model(rf_model, get_bow(t, sub_x_test), sub_y_test, False)
                np.savetxt(f'rf_train_{train_cat}_test_{test_cat}.npy', rf_y_pred)
                rf_results[i] = rf_result

                tmp = np.copy(sub_x_test)
                nn_sub_x_test = get_vecs(tmp, dictionairy, vocab_length, 'last')
                nn_result, nn_y_pred = eval_model(nn_model, nn_sub_x_test, sub_y_test, True)
                np.savetxt(f'nn_train_{train_cat}_test_{test_cat}.npy', nn_y_pred)
                nn_results[i] = nn_result
            return (nn_results, rf_results)

        def eval_model(model, x_test, y_test, nn):
            if nn:
                y_pred = model.predict_classes(x_test)
                result = eval(y_test, y_pred)
            else:
                y_pred = model.predict(x_test)
                result = eval(y_test, y_pred)
            return result[-1], y_pred

        uniq_cats = np.unique(categories)
        category_x_train = [x_train[train_cats == cat] for cat in uniq_cats]
        category_x_test = [x_test[test_cats == cat] for cat in uniq_cats]
        category_y_train = [y_train[train_cats == cat] for cat in uniq_cats]
        category_y_test = [y_test[test_cats == cat] for cat in uniq_cats]

        nn_accuracy_matrix = np.empty((len(uniq_cats), len(uniq_cats)))
        rf_accuracy_matrix = np.empty((len(uniq_cats), len(uniq_cats)))
        for i in range(len(uniq_cats)):
            print(i)
            nn_results, rf_results = train_eval(category_x_train[i], category_y_train[i], category_x_test, category_y_test, uniq_cats, uniq_cats[i])
            nn_accuracy_matrix[i] = nn_results
            rf_accuracy_matrix[i] = rf_results
        return nn_accuracy_matrix, rf_accuracy_matrix, uniq_cats

    def train_eval_all_topics(x_train, y_train, x_test, y_test, val_size, gensim_model, top_tokens):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=val_size)
        np.savetxt('all_true', y_test)

        print('Training a random forest classifier...')
        rf_model, rf_y_pred = perform_rf(x_train, y_train, x_val, y_val, x_test, y_test)
        print('Done!')
        rf_results = eval(y_test, rf_y_pred)
        np.savetxt('rf_train_all_test_all.npy', rf_y_pred)

        #print(f'Getting top {top_tokens} tokens')
        #x_train, x_val, x_test = get_top_tokens(x_train, x_val, x_test, top_tokens)
        #print('Done!')
        x_train, x_val, x_test, emb_layer, dictionairy, vocab_length = get_embedding_weights(x_train, x_val, x_test, gensim_model, 'last')
        max_x = np.amax(x_train)
        print('Training a neural classifier...')
        nn_model = do_lstm(emb_layer, x_train, y_train, x_val, y_val, max_x, 3)
        print('Done!')
        nn_y_pred = nn_model.predict_classes(x_test)
        nn_results = eval(y_test, nn_y_pred)
        np.savetxt('nn_train_all_test_all.npy', nn_y_pred) 
        return (nn_results, rf_results)
    
    all_nn_results, all_rf_results = train_eval_all_topics(original_x_train, original_y_train, x_test, y_test, val_size, gensim_model, top_tokens)
    nn_acc_matrix, rf_acc_matrix, uniq_cats = train_eval_sub_topics(original_x_train, original_y_train, x_test, y_test, val_size, gensim_model, top_tokens)
    np.save('uniq_cats.npy', uniq_cats)

    print('Results for training and testing on all topics:')
    print('For neural network:')
    print(all_nn_results)
    print('For random forest:')
    print(all_rf_results)

    print('Topics are:')
    print(uniq_cats)
    print('Accuracy matrix for training and testing on combinations of topics:')
    print('For neural network:')
    print(nn_acc_matrix)
    print('For random forest:')
    print(rf_acc_matrix)

def wrong_question_answer_pairs(pred_file, true_file, questions, answers, question_ids):

    def int_to_label(int_labels):
        def mapping(x):
            if x == 0:
                return 'yes'
            elif x == 1:
                return 'no'
            elif x == 2:
                return 'na'
            else:
                raise Exception(f'Can not map int {x} to label')
        text_labels = np.vectorize(mapping)(int_labels)
        return text_labels

    v_int = np.vectorize(int)
    test_ids = v_int(np.loadtxt('WS/testing_ids.txt'))
    question_ids = question_ids.tolist()
    test_idxs = [question_ids.index(test_id) for test_id in test_ids]
    test_questions = questions[test_idxs]
    test_answers = answers[test_idxs]
    true_arr = np.loadtxt(true_file)
    pred_arr = np.loadtxt(pred_file)
    wrong_questions = test_questions[pred_arr != true_arr]
    wrong_answers = test_answers[pred_arr != true_arr]
    pred_labels = int_to_label(pred_arr[pred_arr != true_arr])
    true_labels = int_to_label(true_arr[pred_arr != true_arr])
    wrong_pairs = np.stack((wrong_questions, wrong_answers, pred_labels, true_labels), axis=1)
    np.savetxt('wrong_pairs.csv', wrong_pairs, delimiter=',', fmt='%s', encoding='utf-8', newline='\n')
    return (None)

def quality_error_analysis(pred_file, true_file, questions, answers, question_ids, answer_qualities):
    v_int = np.vectorize(int)
    v_len = np.vectorize(len)
    test_ids = v_int(np.loadtxt('WS/testing_ids.txt'))
    question_ids = question_ids.tolist()
    test_idxs = [question_ids.index(test_id) for test_id in test_ids]
    test_qualities = answer_qualities[test_idxs]
    test_answers = answers[test_idxs]
    true_arr = np.loadtxt(true_file)
    pred_arr = np.loadtxt(pred_file)
    wrong_qualities = test_qualities[pred_arr != true_arr]
    correct_qualities = test_qualities[pred_arr == true_arr]
    wrong_length = v_len(sanitize(test_answers[pred_arr != true_arr]))
    correct_length = v_len(sanitize(test_answers[pred_arr == true_arr]))
    wrong_mean_length = np.mean(wrong_length)
    correct_mean_length = np.mean(correct_length)
    wrong_std_length = np.std(wrong_length)
    correct_std_length = np.std(correct_length)
    mean_wrong = np.mean(wrong_qualities)
    mean_correct = np.mean(correct_qualities)
    std_wrong = np.std(wrong_qualities)
    std_correct = np.std(correct_qualities)
    print(f'Mean answer quality for wrong predictions: {mean_wrong}')
    print(f'Std. of answer quality for wrong predictions: {std_wrong}')
    print(f'Mean length of answer for wrong predictions {wrong_mean_length}')
    print(f'Std. of length of answer for wrong predictions {wrong_std_length}')
    print(f'Mean answer quality for correct predictions: {mean_correct}')
    print(f'Std. of answer quality for correct predictions: {std_correct}')
    print(f'Mean length of answer for correct predictions: {correct_mean_length}')
    print(f'Std. of length of answer for correct predictions {correct_std_length}')
    return (None)

if __name__ == '__main__':
    ###############################
    ### Load & preprocess data ####
    ###############################
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
    test_and_val_size = 0.1

    print('Performing general preprocessing...')
    tokens = sanitize(questions)
    print('Done!')
    preds = np.empty((nn_experiments,nn_metrics))
    for i in range(nn_experiments):
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(tokens, category_ids, test_and_val_size)
        preds[i] = perform_nn(x_train, y_train, x_val, y_val, x_test, y_test, gensim_model)
    print('Average metrics of 10 best NN models:', np.mean(preds, axis=0))
    print('Std. of metrics of 10 best NN models:', np.std(preds, axis=0))

    perform_rf(x_train, y_train, x_val, y_val, x_test, y_test)

    ##############################
    ######## Crowdsourcing #######
    ##############################
    clean_crowdsource()
    questions, question_ids, categories, answers, answer_qualities, answer_labels = load_crowdsource()
    perform_cs_nn(question_ids, answers, answer_qualities, gensim_model, 'first')
    perform_cs_nn(question_ids, answers, answer_qualities, gensim_model, 'last')

    ##############################
    ######## Recommender #########
    ##############################

    perform_recommend_search(questions, question_ids, categories)

    ##############################
    ###### Answer Detector #######
    ##############################
    top_tokens = 750
    perform_answer_detection(question_ids, answers, answer_labels, categories, gensim_model, top_tokens)
    wrong_question_answer_pairs('rf_train_all_test_all.npy', 'all_true', questions, answers, question_ids)
    quality_error_analysis('rf_train_all_test_all.npy', 'all_true', questions, answers, question_ids, answer_qualities)