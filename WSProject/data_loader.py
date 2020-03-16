import os
import jsonlines as jl
import numpy as np
import pandas as pd
import re
import keras
import gensim
import krippendorff
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from gensim.test.utils import datapath

def get_length(docs):
    """
    docs: the documents:\n
    returns: the length of the longest document\n
    """
    length = 0
    for doc in docs:
        doc_len = len(doc)
        length = doc_len if doc_len > length else length
    return length

def word2index(vocab):
    """
    Computes a mapping from\\
    words in a vocabulary to\\
    an index
    Parameters:\\
    vocab: The vocabulary as a set
    Returns:\\
    A dictionairy of word:index
    """
    return {word:i + 1 for i, word in enumerate(vocab)}

def get_vecs(docs, dictionairy, length, first_or_last_tokens='first'):
    """
    Takes as input documents and\\
    calculates their indexes in\\
    the word2vec model
    Parameters:\\
    docs: The documents as a list\\
    of lists of words\\
    dictionairy: word:index of words\\
    in the reviews\\
    length: The length to pad sequences
    Returns:\\
    A list of lists of indexes
    """
    system_max = 750
    docs_idxs = [[] for i in range(len(docs))]
    for i in range(len(docs)):
        j = 0
        docs[i] = np.flip(docs[i]) if first_or_last_tokens == 'last' else docs[i]
        for word in docs[i]:
            if j < system_max:
                try:
                    docs_idxs[i].append(dictionairy[word])
                    j += 1
                except:
                    docs_idxs[i].append(0)
                    j += 1
    return keras.preprocessing.sequence.pad_sequences(docs_idxs, maxlen=min(length, system_max), padding='pre')

def get_vocab(docs):
    """
    Creates a vocabulary of the\\
    words used from the documents
    Parameters:\\
    docs: The questions. Must be\\
    a list of lists of strings
    Returns:\\
    The vocabulary as a set
    """
    flat_lst = [item for sublist in docs for item in sublist]
    return set(flat_lst)

def get_embedding_matrix(dictionairy, model, vector_dim):
    """
    Given a dictionairy and a gensim\\
    model, calculate an embedding matrix\\
    based on the words and their indices\\
    in the dictionairy as well as the\\
    gensim word embedding of the words
    Parameters:\\
    dictionairy: word:i representation\\
    of the vocabulary.\\
    model: The gensim word2vec model\\
    vector_dim: dimensionality of the word embeddings
    Returns:\\
    An embedding matrix
    """
    length = len(dictionairy) + 1
    embedding_matrix = np.zeros((length, vector_dim))
    for word, i in dictionairy.items():
        if word in model.vocab:
            embedding_matrix[i] = model.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print('Non null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) != 0))
    return embedding_matrix

def get_embedding_weights(train_docs, val_docs, test_docs, gensim_model, first_or_last_tokens='first'):
    vocab = get_vocab(train_docs)
    word2index_dict = word2index(vocab)
    length = get_length(train_docs)
    train_vecs = get_vecs(train_docs, word2index_dict, length, first_or_last_tokens)
    val_vecs = get_vecs(val_docs, word2index_dict, length, first_or_last_tokens)
    test_vecs = get_vecs(test_docs, word2index_dict, length, first_or_last_tokens)
    vector_dim = 300
    embedding_matrix = get_embedding_matrix(word2index_dict, gensim_model, vector_dim)
    return (train_vecs, val_vecs, test_vecs, embedding_matrix, word2index_dict, length)

def sanitize(documents):
    """
    documents: array of strings\n
    returns: array of lists of words that have been sanitized\n
    """
    stop_words = set(stopwords.words('english'))
    v_sub = np.vectorize(re.sub)
    text = np.char.lower(documents)
    text = v_sub(r'(\W|[0-9]|_)+', ' ', text)
    tokens = np.char.split(text)
    tokens = np.array([[token for token in lst if token not in stop_words] for lst in tokens])
    return (tokens)



def fit_tokenizer(tokens):
    """
    tokens: array of list of lists of tokens\n
    returns: tokenizer with internal vocabulary fitted on tokens\n
    """
    t = Tokenizer()
    t.fit_on_texts(tokens)
    return (t)

def get_bow(tokenizer, tokens):
    """
    tokenizer: a tokenizer that has been fitted\n
    tokens: array of list of lists of tokens\n
    returns: no_docs * size_vocab matrix of bag of words\n
    """
    bow = tokenizer.texts_to_matrix(tokens,  mode='count')[..., 1:]
    return (bow)


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
    uniq_counts = [np.sum(ids == id) for id in uniq_ids]
    mean_q_lens = [np.mean(t_q_len) for t_q_len in [q_len[ids == topic] for topic in uniq_ids]]
    mean_a_lens = [np.mean(t_a_len) for t_a_len in [a_len[ids == topic] for topic in uniq_ids]]
    std_q_lens = [np.std(t_q_len) for t_q_len in [q_len[ids == topic] for topic in uniq_ids]]
    std_a_lens = [np.std(t_a_len) for t_a_len in [a_len[ids == topic] for topic in uniq_ids]]
    return (uniq_cats, uniq_ids, uniq_counts, mean_q_lens, mean_a_lens, std_q_lens, std_a_lens)



def load_data(path, labels=False):
    """
    path: path to jsonl file\n
    returns: three numpy arrays of questions, answers, and categoryIds\n
    """
    with jl.open(path) as reader:
       if labels:
           vals = np.asarray([[obj['question'], obj['answer'], obj['categoryId'], obj['category'], obj['questionId'], obj['answerLabel'], obj['answerQuality']] for obj in reader])
       else:
           vals = np.asarray([[obj['question'], obj['answer'], obj['categoryId'], obj['category']] for obj in reader])
       v_int = np.vectorize(int)
       questions = vals[..., 0]
       answers = vals[..., 1]
       category_ids = v_int(vals[..., 2])
       categories = vals[..., 3]
       if labels:
           question_ids = v_int(vals[..., 4])
           answer_labels = vals[..., 5]
           answer_qualities = v_int(vals[..., 6])
           return (questions, answers, category_ids, categories, question_ids, answer_labels, answer_qualities)
       else:
           return (questions, answers, category_ids, categories)




def load_cs_csvs(path):
    """
    path: path to the directory containing the csv files\n
    returns: a single pandas dataframe resulting from concatenating the loaded csv files\n
    """
    csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
    return pd.concat([pd.read_csv(path + '/' + file, error_bad_lines=False, warn_bad_lines=True) for file in csv_files], ignore_index=True)

def mv_tiebreaker(ans):
    """
    Takes as input an array of answers and outputs the majority vote. Ties are broken by a random choice between the tie makers.\n
    """
    uniq_ans, counts = np.unique(ans, return_counts=True)
    max_vals = uniq_ans[counts == np.max(counts)]
    tie_breaker = np.random.randint(0, len(max_vals))
    res = max_vals[tie_breaker]
    return (res)

def mv_average(ans):
    """
    Takes as input an array of answers and outputs the majority vote. Ties are broken by an average between the tie makers.\n
    """
    uniq_ans, counts = np.unique(ans, return_counts=True)
    max_vals = uniq_ans[counts == np.max(counts)]
    return (np.average(max_vals))

def clean_and_stats_dataframe(df):
    rgx = r'[^\n]*\n.'
    df['Question'].replace(to_replace=rgx, value='', inplace=True, regex=True) # Remove the broken values from WS1002.csv
    df['Answer Label'] = df['Answer Label'].astype(str).apply(str.strip).apply(str.lower)
    df['Answer Label'].replace(to_replace='nan', value='na', inplace=True)
    new_df = pd.DataFrame(columns=df.columns)
    list_of_series = []
    non_facts = 0
    total_a_labels = []
    total_q_ratings = []
    total_a_quals = []
    print('Cleaning dataset...')
    for q_id in np.unique(df['questionId']):
        sub_df = df.loc[df['questionId'] == q_id]

        facts = sub_df['Factual'].to_numpy(dtype=float)
        mv_fact = mv_tiebreaker(facts)

        # Ignore non-factual questions
        if mv_fact != 1:
            non_facts += 1
            continue

        question, a_url = sub_df.iloc[0,:-5]

        a_labels = sub_df['Answer Label'].to_numpy(dtype=str)
        q_ratings = sub_df['Question Rating'].to_numpy(dtype=float)
        a_quals = sub_df['Answer Quality'].to_numpy(dtype=float)

        total_a_labels.append(a_labels.tolist())
        total_q_ratings.append(q_ratings.tolist())
        total_a_quals.append(a_quals.tolist())

        mv_a_label = mv_tiebreaker(a_labels)
        mv_q_rating = mv_average(q_ratings)
        mv_a_qual = mv_average(a_quals)

        series = pd.Series([question, a_url, mv_a_label, mv_q_rating, mv_a_qual, mv_fact, q_id], index=new_df.columns)
        list_of_series.append(series)
        # end for
    print('Done!')
    new_df = new_df.append(list_of_series, ignore_index=True)

    print('Computing some statistics...')

    total_a_labels = [item for sublist in total_a_labels for item in sublist]
    total_q_ratings = [item for sublist in total_q_ratings for item in sublist]
    total_a_quals = [item for sublist in total_a_quals for item in sublist]
    mean_q_rating = np.mean(total_q_ratings)
    std_q_rating = np.std(total_q_ratings)
    mean_a_qual = np.mean(total_a_quals)
    std_a_qual = np.std(total_a_quals)

    total_qrs = len(total_q_ratings)
    total_aqs = len(total_a_quals)

    qr_ones = total_q_ratings.count(1.0)
    qr_twos = total_q_ratings.count(2.0)
    qr_threes = total_q_ratings.count(3.0)

    aq_ones = total_a_quals.count(1.0)
    aq_twos = total_a_quals.count(2.0)
    aq_threes = total_a_quals.count(3.0)

    al_yes = total_a_labels.count('yes')
    al_no = total_a_labels.count('no')
    al_na = total_a_labels.count('na')

    cleaned_qr_ones = len(new_df.loc[new_df['Question Rating'] == 1])
    cleaned_qr_one_fives = len(new_df.loc[new_df['Question Rating'] == 1.5])
    cleaned_qr_twos = len(new_df.loc[new_df['Question Rating'] == 2])
    cleaned_qr_two_fives = len(new_df.loc[new_df['Question Rating'] == 2.5])
    cleaned_qr_threes = len(new_df.loc[new_df['Question Rating'] == 3])

    cleaned_aq_ones = len(new_df.loc[new_df['Answer Quality'] == 1])
    cleaned_aq_one_fives = len(new_df.loc[new_df['Answer Quality'] == 1.5])
    cleaned_aq_twos = len(new_df.loc[new_df['Answer Quality'] == 2])
    cleaned_aq_two_fives = len(new_df.loc[new_df['Answer Quality'] == 2.5])
    cleaned_aq_threes = len(new_df.loc[new_df['Answer Quality'] == 3])

    cleaned_al_yes = len(new_df.loc[new_df['Answer Label'] == 'yes'])
    cleaned_al_no = len(new_df.loc[new_df['Answer Label'] == 'no'])
    cleaned_al_na = len(new_df.loc[new_df['Answer Label'] == 'na'])

    cleaned_qrs = new_df['Question Rating'].to_numpy(dtype=float)
    mean_cleaned_qr = np.mean(cleaned_qrs)
    std_cleaned_qr = np.std(cleaned_qrs)

    cleaned_aqs = new_df['Answer Quality'].to_numpy(dtype=float)
    mean_cleaned_aq = np.mean(cleaned_aqs)
    std_cleaned_aq = np.std(cleaned_aqs)

    print('Computing Krippendorffs Alpha...')
    krippen_alpha_fact = krippen_alpha(df, 'Factual', 'nominal')
    krippen_alpha_qr = krippen_alpha(df, 'Question Rating', 'interval')
    krippen_alpha_aq = krippen_alpha(df, 'Answer Quality', 'interval')
    krippen_alpha_al = krippen_alpha(df, 'Answer Label', 'nominal')
    print()
    print()
    print('Statistics for segregated dataset:')
    print()
    print('Statistics for question ratings:')
    print('Mean question rating:', mean_q_rating)
    print('Std. of question rating:', std_q_rating)
    print('Number of question rating 1\'s:', qr_ones)
    print('Number of question rating 2\'s:', qr_twos)
    print('Number of question rating 3\'s:', qr_threes)
    print()
    print('Statistics for answer qualities:')
    print('Mean answer quality:', mean_a_qual)
    print('Std. of answer quality:', std_a_qual)
    print('Number of answer rating 1\'s:', aq_ones)
    print('Number of answer rating 2\'s:', aq_twos)
    print('Number of answer rating 3\'s:', aq_threes)
    print()
    print('Statistics for answer labels:')
    print('Number yes-labels given', al_yes)
    print('Number of no-labels given', al_no)
    print('Number of na-labels given', al_na)
    print()
    print('Krippendorff\'s alpha coefficients for:')
    print('Factual:', krippen_alpha_fact)
    print('Question Rating:', krippen_alpha_qr)
    print('Answer Quality:', krippen_alpha_aq)
    print('Answer Label:', krippen_alpha_al)
    print()
    print()
    print('Statistics for aggregated dataset:')
    print()
    print('Number of non-factual questions that have been disregarded in the following:', non_facts)
    print('Mean question rating:', mean_cleaned_qr)
    print('Std. of question rating:', std_cleaned_qr)
    print('Number of question rating 1\'s:', cleaned_qr_ones)
    print('Number of question rating 1.5\'s:', cleaned_qr_one_fives)
    print('Number of question rating 2\'s:', cleaned_qr_twos)
    print('Number of question rating 2.5\'s:', cleaned_qr_two_fives)
    print('Number of question rating 3\'s:', cleaned_qr_threes)
    print()
    print('Statistics for answer qualities:')
    print('Mean answer quality:', mean_cleaned_aq)
    print('Std. of answer quality:', std_cleaned_aq)
    print('Number of answer rating 1\'s:', cleaned_aq_ones)
    print('Number of answer rating 1.5\'s:', cleaned_aq_one_fives)
    print('Number of answer rating 2\'s:', cleaned_aq_twos)
    print('Number of answer rating 2.5\'s:', cleaned_aq_two_fives)
    print('Number of answer rating 3\'s:', cleaned_aq_threes)
    print()
    print('Statistics for answer labels:')
    print('Number yes-labels given', cleaned_al_yes)
    print('Number of no-labels given', cleaned_al_no)
    print('Number of na-labels given', cleaned_al_na)
    return

def krippen_alpha(df, metric, lom):
    """
    Given a dataframe, a column in that dataframe, and a level of measurement, compute Krippendorff's Alpha for that column and level of measurement.
    """
    metric_vals = np.unique(df[metric])

    def metric_vals_to_indices(vals, mvals):
        indices = [np.nonzero(mvals == val)[0][0] for val in vals]
        return np.array(indices)

    uniq_ids = np.unique(df['questionId'])
    shape = (uniq_ids.shape[0], metric_vals.shape[0])
    value_counts = np.zeros(shape, dtype=float)
    for i, q_id in zip(range(len(uniq_ids)), uniq_ids):
        assignments = df.query('questionId==@q_id')[metric].to_numpy()
        vals, counts = np.unique(assignments, return_counts=True)
        indices = metric_vals_to_indices(vals, metric_vals)
        value_counts[i][indices] += counts
    alpha = krippendorff.alpha(value_counts=value_counts, value_domain=metric_vals.tolist(), level_of_measurement=lom)
    return alpha