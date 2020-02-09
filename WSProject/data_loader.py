import jsonlines as jl
import numpy as np
import pandas as pd


def stats(questions, answers, cats, ids):
    """
    questions: array of questions\n
    answers: array of answers\n
    cats: array of categories\n
    ids: array of ids\n
    returns: some statistics of the questions and answers based on their category
    """
    v_len = np.vectorize(len)
    q_len = v_len(questions)
    a_len = v_len(answers)
    uniq_cats = pd.unique(cats)
    uniq_ids = pd.unique(ids)
    uniq_counts = [np.sum(ids==id) for id in uniq_ids]
    mean_q_lens = [np.mean(t_q_len) for t_q_len in [q_len[ids==topic] for topic in uniq_ids]]
    mean_a_lens = [np.mean(t_a_len) for t_a_len in [a_len[ids==topic] for topic in uniq_ids]]
    std_q_lens = [np.std(t_q_len) for t_q_len in [q_len[ids==topic] for topic in uniq_ids]]
    std_a_lens = [np.std(t_a_len) for t_a_len in [a_len[ids==topic] for topic in uniq_ids]]
    return (uniq_cats, uniq_ids, uniq_counts, mean_q_lens, mean_a_lens, std_q_lens, std_a_lens)



def load_data(path):
    """
    path: path to jsonl file\n
    returns: three numpy arrays of questions, answers, and categoryIds\n
    """
    with jl.open(path) as reader:
       vals = np.asarray([[obj['question'], obj['answer'], obj['categoryId'], obj['category']] for obj in reader])
       questions = vals[..., 0]
       answers = vals[..., 1]
       v_int = np.vectorize(int)
       category_ids = v_int(vals[..., 2])
       categories = vals[..., 3]
       return (questions, answers, category_ids, categories)