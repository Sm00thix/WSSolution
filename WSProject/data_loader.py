import jsonlines as jl
import numpy as np
import pandas as pd
import re
import keras
import gensim
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec

def get_length(docs):
    """
    docs: the documents:\n
    returns: the length of the longest document\n
    """
    length = 0
    #for docset in docs:
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
    reviews: The reviews as a list\\
    of lists of words\\
    dictionairy: word:index of words\\
    in the reviews\\
    length: The length to pad sequences
    Returns:\\
    A list of lists of indexes
    """
    #docs_idxs = [[dictionairy[word] for word in doc] for doc in docs]

    docs_idxs = [[] for i in range(len(docs))]
    for i in range(len(docs)):
        for word in docs[i]:
            try:
                docs_idxs[i].append(dictionairy[word])
            except:
                docs_idxs[i].append(0)

    #reviews_idxs = [[] for i in range(len(reviews))]
    #for i in range(len(reviews)):
    #    j = 0
    #    for word in reviews[i]:
    #        if j < 250:
    #            reviews_idxs[i].append(dictionairy[word])
    #            j += 1
    return keras.preprocessing.sequence.pad_sequences(docs_idxs, maxlen=length)

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
            #embedding_matrix[i] = model[word]
            embedding_matrix[i] = model.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print('Non null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) != 0))
    return embedding_matrix

def get_embedding_weights(train_docs, test_docs):
    vocab = get_vocab(train_docs)
    word2index_dict = word2index(vocab)
    length = get_length(train_docs) # all_docs must include test set in order to perform the padding correctly
    train_vecs = get_vecs(train_docs, word2index_dict, length)
    test_vecs = get_vecs(test_docs, word2index_dict, length)
    vector_dim = 300
    print("loading word embeddings...")
    gensim_model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Batma/Desktop/WS/GoogleNews-vectors-negative300.bin', binary=True)
    #gensim_model = Word2Vec(train_docs, size=vector_dim, sg=1, min_count=1, window=20, workers=-1)
    #gensim_model.train(train_docs, total_examples=len(train_docs), epochs=20)
    #wv = gensim_model.wv
    embedding_matrix = get_embedding_matrix(word2index_dict, gensim_model, vector_dim)
    return (train_vecs, test_vecs, embedding_matrix)

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