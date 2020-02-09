from data_loader import load_data, stats, sanitize
if __name__ == '__main__':
    ################################
    #### Load & preprocess data ####
    ################################
    questions, answers, category_ids, categories = load_data('WS/web_science_dataset.jsonl')
    unique_categories, unique_ids, unique_counts, mean_q_len, mean_a_len, std_q_len, std_a_len = stats(questions, answers, categories, category_ids)
    
    print('Unique categories:', unique_categories)
    print('Unique ids:', unique_ids)
    print('Counts:', unique_counts)
    print('Mean question length:', mean_q_len)
    print('Mean answer length:', mean_a_len)
    print('Standard deviation of question length:', std_q_len)
    print('Standard deviation of answer length:', std_a_len)
    
    # TODO: Split in to training, validation, and test set - maybe use 5CV?
    # TODO: Remove stop words?
    # TODO: Use BoW on training questions (not answers?) and train random forest and NN to predict topic
    sanitize(questions)
