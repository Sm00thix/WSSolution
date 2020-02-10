from data_loader import *
from classifiers import *
if __name__ == '__main__':
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
    
    tokens = sanitize(questions)
    bow = BoW(tokens)
    x_train, y_train, x_test, y_test = split_dataset(bow, category_ids)

    print('Performing random forest classification...')
    best_params, train_f1, test_f1 = do_random_forest(x_train, y_train, x_test, y_test)
    print('Best parameters for random forest:', best_params)
    print('Training f1 score for random forest', train_f1)
    print('Test f1 score for random forest', test_f1)