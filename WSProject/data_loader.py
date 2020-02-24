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
    return {word:i+1 for i, word in enumerate(vocab)}

def get_vecs(docs, dictionairy, length):
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
    docs_idxs = [[] for i in range(len(docs))]
    for i in range(len(docs)):
        for word in docs[i]:
            try:
                docs_idxs[i].append(dictionairy[word])
            except:
                docs_idxs[i].append(0)
    return keras.preprocessing.sequence.pad_sequences(docs_idxs, maxlen=length, padding='pre')

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
    based on the words and ther indices\\
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

def get_embedding_weights(train_docs, val_docs, test_docs):
    vocab = get_vocab(train_docs)
    word2index_dict = word2index(vocab)
    length = get_length(train_docs)
    train_vecs = get_vecs(train_docs, word2index_dict, length)
    val_vecs = get_vecs(val_docs, word2index_dict, length)
    test_vecs = get_vecs(test_docs, word2index_dict, length)
    vector_dim = 300
    print("Loading word embeddings...")
    filename = 'WS/GoogleNews-vectors-negative300'
    try:
        gensim_model = gensim.models.KeyedVectors.load(filename, mmap='r')
    except:
        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(filename + '.bin', binary=True)
        gensim_model.wv.save(filename)

    print("Done loading word embeddings!")
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

def load_cs_csvs(path):
    """
    path: path to the directory containing the csv files\n
    returns: a single pandas dataframe resulting from concatenating the loaded csv files\n
    """
    csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
    lol = pd.read_csv(path+'/'+'WS1002.csv')
    return pd.concat([pd.read_csv(path+'/'+file, error_bad_lines=False, warn_bad_lines=True) for file in csv_files], ignore_index=True)

def majority_vote(ans):
    """
    Takes as input an array of answers and outputs the majority vote. Ties are broken by a random choice between the tie makers.\n
    """
    uniq_ans, counts = np.unique(ans, return_counts=True)
    max_vals = uniq_ans[counts == np.max(counts)]
    tie_breaker = np.random.randint(0, len(max_vals))
    res = max_vals[tie_breaker]
    return (res)

def clean_dataframe(df):
    rgx = r'[^\n]*\n.'
    df['Question'].replace(to_replace=rgx, value='', inplace=True, regex=True) # Remove the broken values from WS1002.csv
    v_strip = np.vectorize(str.strip)
    v_lower = np.vectorize(str.lower)
    new_df = pd.DataFrame(columns=df.columns)
    list_of_series = []
    non_facts = 0
    total_a_labels = []
    total_q_ratings = []
    total_a_quals = []
    for q_id in np.unique(df['questionId']):
        sub_df = df.loc[df['questionId'] == q_id]

        facts = sub_df['Factual'].to_numpy(dtype=float)
        mv_fact = majority_vote(facts)

        # Ignore non-factual questions
        if mv_fact != 1:
            non_facts += 1
            continue

        question, a_url = sub_df.iloc[0,:-5]

        a_labels = v_strip(v_lower(sub_df['Answer Label'].to_numpy(dtype=str)))
        q_ratings = sub_df['Question Rating'].to_numpy(dtype=float)
        a_quals = sub_df['Answer Quality'].to_numpy(dtype=float)

        total_a_labels.append(a_labels)
        total_q_ratings.append(q_ratings)
        total_a_quals.append(a_quals)

        mv_a_label = majority_vote(a_labels)
        mv_q_rating = majority_vote(q_ratings)
        mv_a_qual = majority_vote(a_quals)

        series = pd.Series([question, a_url, mv_a_label, mv_q_rating, mv_a_qual, mv_fact, q_id], index=new_df.columns)
        list_of_series.append(series)
    new_df = new_df.append(list_of_series, ignore_index=True)
    return
    # use old df for reliability rating
    # perform some stats - use the lists as before-mv and use new_df as after-mv