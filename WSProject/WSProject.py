from data_loader import *
from classifiers import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import pandas as pd
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

def perform_recommend(questions, question_ids, categories, train_df, test_df, k):

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

    # split questions into categories
    #q_nutrition = questions[categories == 'nutrition']
    #q_climate_change = questions[categories == 'climate-change']
    #q_medical_science = questions[categories == 'medical-science']
    #q_physics = questions[categories == 'physics']
    #q_psychology = questions[categories == 'psychology']

    # keep track of question ids in the categorical splits
    #id_nutrition = question_ids[categories == 'nutrition']
    #id_climate_change = question_ids[categories == 'climate-change']
    #id_medical_science = question_ids[categories == 'medical-science']
    #id_physics = question_ids[categories == 'physics']
    #id_psychology = question_ids[categories == 'psychology']

    # compute tfidf
    tokens = sanitize(questions)
    tfidf = get_tf_idf(tokens)
    #tfidf_nutrition = tfidf[categories == 'nutrition']
    #tfidf_climate_change = tfidf[categories == 'climate-change']
    #tfidf_medical_science = tfidf[categories == 'medical-science']
    #tfidf_physics = tfidf[categories == 'physics']
    #tfidf_psychology = tfidf[categories == 'psychology']

    users = pd.unique(train_df['userID'])
    vector_len = tfidf.shape[1]

    def compute_scores(train_category, test_category, k):
        train_question_ids = question_ids[categories == train_category]
        test_question_ids = question_ids[categories == test_category]
        # must split query in two to avoid scoping issues
        sub_train_df = train_df.query('questionID in @train_question_ids')
        users_likes = [sub_train_df.query('userID == @user and rating == 3')['questionID'] for user in users]
        sub_recommend_df = train_df.query('questionID in @test_question_ids')
        users_recommends = np.array([sub_recommend_df.query('userID == @user and rating == 3')['questionID'].to_numpy() for user in users])
        users_not_recommends = np.array([sub_recommend_df.query('userID == @user and rating == 1')['questionID'].to_numpy() for user in users])
        #sub_test_df = test_df.query('questionID in @test_category_ids')
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
            friends = users[np.argsort(cos_sims[i])[::-1][1:k+1]]
            top_k_friends[i] = friends
            friends_idxs = [np.where(users==friend)[0][0] for friend in friends]
            friends_recommends = users_recommends[friends_idxs]
            friends_not_recommends = users_not_recommends[friends_idxs]
            recommendations[i].append(aggregate_recommends(friends_recommends, friends_not_recommends))
        # recommendations have now been computed.
        # compute the accuracy between recommendations and actual recommended items (TP + TN) / (TP + TN + FP + FN), TN is any "no" in the test set that was not recommended.
        # Save result in the result matrix
        pass
    uniq_cats = np.unique(categories)
    accuracy_matrix = np.empty((len(uniq_cats), len(uniq_cats)))
    for i in range(len(uniq_cats)):
        for j in range(len(uniq_cats)):
            accuracy_matrix[i] = compute_scores(uniq_cats[i], uniq_cats[j], k)
            print('Done for train: ', uniq_cats[i], ' and test: ', uniq_cats[j])
    pass

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
    k_friends = 5
    train_df, test_df = load_tsvs('WS/')
    perform_recommend(questions, question_ids, categories, train_df, test_df, k_friends)