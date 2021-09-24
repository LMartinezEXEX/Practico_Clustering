import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.cluster import kmeans, cosine_distance
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

FILE_NAME = "Corpus/LaVanguardia.txt"
MIN_FREQUENCY = 10
BACKWARD_WINDOW = 2
FORWARD_WINDOW = 2

def get_dataset(size=0):
    '''Get file content from FILE_NAME'''

    file = open(FILE_NAME, 'r', encoding='latin-1')
    dataset = ''
    if size:
        dataset = file.read(size)    
    else:
        dataset = file.read()
    file.close()
    return dataset

def spacy_processing(dataset):
    '''Process dataset with regular Spacy pipeline'''

    nlp = spacy.load('es_core_news_lg')
    doc = nlp(dataset)
    return doc

def get_tokens(doc):
    '''Get useful Spacy Tokens from a Spacy Doc object'''
    tokens = []
    for sent in doc.sents:
        for token in sent:
            if token.is_alpha:
                tokens.append(token)

    return tokens

def get_word_frequency(words):
    '''Get words frequency from a Spacy Token object array'''

    words_freq = {}
    counter = Counter([word.lemma_ for word in words])
    for word in counter:
        if counter[word] >= MIN_FREQUENCY:
            words_freq[word] = counter[word]

    return words_freq

def get_words_vectors(doc, tokens, words_freq):
    '''Create vectors based on selected features'''

    word_dicc = {}

    for token in tokens:
        lemma = token.lemma_

        if (not token.is_alpha) or token.is_digit or (lemma in words_freq and words_freq[lemma] < MIN_FREQUENCY):
            continue

        if not lemma in word_dicc:
            features = {}
        else:
            features = word_dicc[lemma]

        pos = token.pos_
        if not pos in features:
            features[pos] = 0
        features[pos] += 1

        dep = token.dep_
        if not dep in features:
            features[dep] = 0
        features[dep] += 1

        start = max(0, token.i-BACKWARD_WINDOW)
        end = min(len(doc), token.i + FORWARD_WINDOW + 1)

        for pos in range(start, end):
            co_token = doc[pos]
            if co_token.is_alpha and (co_token.lemma_ in words_freq and words_freq[co_token.lemma_] >= MIN_FREQUENCY):
                if not co_token.lemma_ in features:
                    features[co_token.lemma_] = 0
                features[co_token.lemma_] += 1

        tripla = "TRIPLA__" + lemma + "__" + dep + "__" + token.head.lemma_
        if not tripla in features:
            features[tripla] = 0
        features[tripla] += 1

        word_dicc[lemma] = features

    vectors = []
    word_ids = {}
    wid = 0
    for word in word_dicc:
        if len(word) > 0:
            word_ids[word] = wid
            vectors.append(word_dicc[word])
            wid += 1

    return vectors, word_ids

def get_co_matrix(vectors):
    '''Generate co occurence matrix from the features used'''

    dictVectorizer = DictVectorizer(sparse=False)
    return dictVectorizer.fit_transform(vectors)

def process_matrix(matrix):
    '''Embedding to reduce dimensionality'''

    normed_matrix = matrix / matrix.max(axis=0)
    variances = np.square(normed_matrix).mean(axis=0) - np.square(normed_matrix.mean(axis=0))
    return np.delete(normed_matrix, np.where(variances < 0.0002), axis=1)

def tsne(matrix):
    '''Embedding to reduce dimensionality using T-sne with 2 dimensions'''
    tsne = TSNE(random_state = 2)
    return tsne.fit_transform(matrix)

def LSA(matrix):
    '''Embedding to reduce dimensionality using LSA (single value descomposition)'''
    lsa = TruncatedSVD(n_components=100)
    return lsa.fit_transform(matrix)

def KMeans_clusters(matrix):
    '''Clustering matrix with KMeans algorithm'''

    kmean = kmeans.KMeansClusterer(100, cosine_distance, avoid_empty_clusters=True)
    return kmean.cluster(matrix, True)

def plot_general(pointsspacy):
    pointsspacy.plot.scatter("x", "y", s=10, figsize=(20, 12))

def plot_general_with_clusters(pointscluster):
    pointscluster.plot.scatter(x='x', y='y', c='c', cmap='viridis', s=10, figsize=(20, 12))

def plot_lsa(lsa_matrix):
    xs = [w[0] for w in lsa_matrix]
    ys = [w[1] for w in lsa_matrix]
    plt.scatter(xs, ys)

def plot_region(pointsspacy, x_bounds, y_bounds):
    slice = pointsspacy[
        (x_bounds[0] <= pointsspacy.x) &
        (pointsspacy.x <= x_bounds[1]) & 
        (y_bounds[0] <= pointsspacy.y) &
        (pointsspacy.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter(x='x', y='y', s=35, figsize=(10, 8))
    for _, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


def merge_clusters_values(clusters, word_ids):
    '''Generate a dicc with cluster number as key and list of strings from the words as value'''
    dicc = {}

    for i in range(len(clusters)):
        if clusters[i] in dicc:
            value = dicc[clusters[i]]
            value.append(list(word_ids.keys())[list(word_ids.values()).index(i)])
            dicc[clusters[i]] = value
        else:
            dicc[clusters[i]] = [list(word_ids.keys())[list(word_ids.values()).index(i)]]

    return dicc

def save_clusters(dicc):
    '''Save clusters with its values in txt file'''
    i = 0
    with open('Resultados/iteracion_1_CLUSTERS.txt', 'w') as file:
        for index in dicc:
            file.writelines(["Cluster ", str(i), ":\n"])
            for word in dicc[index]:
                file.writelines([word, ", "])
            file.write('\n\n')
            i += 1

def optimal_k(matrix):
    '''Get best k for K-means with silhouette score'''

    sil = []
    for k in range(2, 150):
        print(k)
        kmeans = KMeans(n_clusters = k).fit(matrix)
        labels = kmeans.labels_
        sil.append(silhouette_score(matrix, labels, metric='euclidean'))

    plt.plot([k for k in range(2, 150)], sil)
    plt.show()

if __name__ == "__main__":
    dataset = get_dataset(200_000)

    doc = spacy_processing(dataset)

    tokens = get_tokens(doc)

    words_freq = get_word_frequency(tokens)

    words_vector, word_ids = get_words_vectors(doc, tokens, words_freq)

    co_matrix = get_co_matrix(words_vector)

    matrix = process_matrix(co_matrix)

    tsne_matrix = tsne(matrix)

    lsa_matrix = LSA(matrix)

    #optimal_k(lsa_matrix)

    clusters = KMeans_clusters(lsa_matrix)

    dicc = merge_clusters_values(clusters, word_ids)

    save_clusters(dicc)

    pointsspacy = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, tsne_matrix[word_ids[word]])
            for word in word_ids
        ]
    ],
    columns=["word", "x", "y"]
    )

    pointscluster = pd.DataFrame(
    [
        (word, coords[0], coords[1], cluster)
        for word, coords, cluster in [
            (word, tsne_matrix[word_ids[word]], clusters[word_ids[word]])
            for word in word_ids
        ]
    ],
    columns=["word", "x", "y", "c"]
    )

    #plot_general(pointsspacy)

    plot_general_with_clusters(pointscluster)

    #plot_lsa(lsa_matrix)

    #plot_region(pointsspacy, (-80, 80), (-70, 70))

    plt.show()