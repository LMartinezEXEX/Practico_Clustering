import re
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from nltk.cluster import kmeans, cosine_distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer

class WordClustering:
    def __init__(self, path_to_dataset, dataset_size = -1) -> None:
        self.path_to_dataset = path_to_dataset
        self.dataset_size = dataset_size

    def get_dataset(self, clean = True):
        '''
        Generate dataset attribute.
        If 'clean' = True then remove signs and numbers.
        '''
        file = open(self.path_to_dataset, 'r', encoding='latin-1')
        dataset = file.read(self.dataset_size)
        file.close()

        if clean:
            dataset = re.sub('&#[0-9]{0,5}.', '', dataset)

        self.dataset = dataset

    def spacy_pipe_processing(self):
        '''
        Generate doc attribute with the dataset given previously.
        '''
        nlp = spacy.load('es_core_news_lg', exclude=['vectors'])
        doc = nlp(self.dataset)

        self.doc = doc

    def get_sents(self, min_frequency = 0):
        '''
        Generate sents attributes with sents longer than 'min_frequency' words.
        '''
        sents = [sent for sent in self.doc.sents if len(sent) > min_frequency]

        self.sents = sents

    def get_tokens(self, source = "DOC"):
        '''
        Generate tokens attribute. It could be either from the doc or filtered sents.
        '''
        tokens = []
        
        if source == "SENTS":
            for sent in self.sents:
                for token in sent:
                    if token.is_alpha:
                        tokens.append(token)
        else:
            for token in self.doc:
                if token.is_alpha:
                    tokens.append(token)

        self.tokens = tokens

    def get_token_frequency(self, min_frequency = 0):
        '''
        Generate tokens lemma frequency attribute from tokens.
        '''
        token_freq = {}
        counter = Counter([token.lemma_ for token in self.tokens])
        for token_lemma in counter:
            if counter[token_lemma] >= min_frequency:
                token_freq[token_lemma] = counter[token_lemma]
        
        self.token_freq = token_freq

    def vector_generator(self, features_to_use = ["POS", "DEP", "ENT", "WIN", "TRI"], window = (2, 2),  min_frequency = 0):
        '''
        Generate list of vectors attribute. Each vector is a representation of a token.
        features_to_use indicates which features use in the representation:
        * POS: use part-of-speech tag.
        * DEP: use dependency tag.
        * ENT: use entity tag if token is an entity.
        * WIN: use the window of co-ocurrence words with window[0] words left to the actual token
        and window[1] words right to the actual token. Must have more than 'min_frequency' ocurrences.
        * TRI: use triple dependency.
        '''
        word_dicc = {}
        for token in self.tokens:
            lemma = token.lemma_

            if (not token.is_alpha) or token.is_digit or (lemma in self.token_freq and self.token_freq[lemma] < min_frequency):
                continue

            if not lemma in word_dicc:
                features = {}
            else:
                features = word_dicc[lemma]

            if "POS" in features_to_use:
                pos = token.pos_
                if not pos in features:
                    features[pos] = 0
                features[pos] += 1

            if "DEP" in features_to_use:
                dep = token.dep_
                if not dep in features:
                    features[dep] = 0
                features[dep] += 1
            
            if "ENT" in features_to_use:
                if token.ent_type:
                    ent = token.ent_type_
                    if not ent in features:
                        features[ent] = 0
                    features[ent] += 1

            if "WIN" in features_to_use:
                start = max(0, token.i-window[0])
                end = min(len(self.doc), token.i + window[1] + 1)

                for position in range(start, end):
                    co_token = self.doc[position]
                    if co_token.is_alpha and (co_token.lemma_ in self.token_freq and self.token_freq[co_token.lemma_] >= min_frequency):
                        if not co_token.lemma_ in features:
                            features[co_token.lemma_] = 0
                        features[co_token.lemma_] += 1

            if "TRI" in features_to_use:
                tripla = "TRIPLA__" + lemma + "__" + dep + "__" + token.head.lemma_
                if not tripla in features:
                    features[tripla] = 0
                features[tripla] += 1

            word_dicc[lemma] = features
        
        vectors = []
        word_ids = {}
        wid = 0
        for word in word_dicc:
            if len(word) > 0 and len(word_dicc[word]) > 0:
                word_ids[word] = wid
                vectors.append(word_dicc[word])
                wid += 1
        
        self.vectors = vectors
        self.word_ids = word_ids

        for vector in vectors:
            if not np.any(vector):
                print(vector)

    def generate_feature_matrix(self, process = True, min_variance = 0.0002):
        '''
        Generate matrix attribute from the vectors.
        If 'process' = True, remove rows with lack of useful information.
        '''
        dv = DictVectorizer(sparse=False)
        matrix = dv.fit_transform(self.vectors)

        if process:
            normed_matrix = matrix / matrix.max(axis=0)
            variances = np.square(normed_matrix).mean(axis=0) - np.square(normed_matrix.mean(axis=0))
            matrix = np.delete(normed_matrix, np.where(variances < min_variance), axis=1)

        self.matrix = matrix

    def apply_TSNE(self, matrix_to_TSNE = "", n_components = 2):
        '''
        Generate tsne attribute from matrix.
        'matrix_to_use' can be "W2V" for word2vec iteration, else it would use the manual generated matrix.
        'n_components' number of components, to reduce dimensions, in T-SNE algorithm.
        Initializes attribute por plotting.
        '''
        tsne = TSNE(n_components = n_components, random_state = 2)
        if matrix_to_TSNE == "W2V":
            tsne_matrix = tsne.fit_transform(self.w2v_matrix)
        else:
            tsne_matrix = tsne.fit_transform(self.matrix)

        self.tsne_matrix = tsne_matrix

        self.pointsspacy = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [(word, self.tsne_matrix[self.word_ids[word]]) for word in self.word_ids]
        ],
        columns=["word", "x", "y"]
        )

    def apply_LSA(self, matrix_to_LSA = "", n_components = 100):
        '''
        Generate lsa attribute from matrix.
        'matrix_to_use' can be "W2V" for word2vec iteration, else it would use the manual generated matrix.
        'n_components' number of components, to reduce dimensions, in LSA algorithm.
        Initializes attribute por plotting.
        '''
        lsa = TruncatedSVD(n_components = n_components)
        if matrix_to_LSA == "W2V":
            lsa_matrix = lsa.fit_transform(self.w2v_matrix)
        else:
            lsa_matrix = lsa.fit_transform(self.matrix)

        self.lsa_matrix = lsa_matrix

    def apply_Kmeans_clustering(self, matrix_to_cluster, n_clusters, distance = cosine_distance):
        '''
        Generate clusters applying K-means algorithm with 'n_clusters' clusters and using 'distance' distance.
        'matrix_to_cluster' could be "LSA", "TSNE", "W2V" or else it would use the manually generated matrix.
        Initializes attribute por plotting.
        '''
        kmean = kmeans.KMeansClusterer(n_clusters, distance, avoid_empty_clusters=True)
        if matrix_to_cluster == "LSA":
            clusters = kmean.cluster(self.lsa_matrix, True)
        elif matrix_to_cluster == "TSNE":
            clusters = kmean.cluster(self.tsne_matrix, True)
        elif matrix_to_cluster == "W2V":
            clusters = kmean.cluster(self.w2v_matrix, True)
        else:
            clusters = kmean.cluster(self.matrix, True)

        self.clusters = clusters

        self.pointscluster = pd.DataFrame(
        [
            (word, coords[0], coords[1], cluster)
            for word, coords, cluster in [(word, self.tsne_matrix[self.word_ids[word]], self.clusters[self.word_ids[word]]) for word in self.word_ids]
        ],
        columns=["word", "x", "y", "c"]
        )

    def plot_points(self):
        '''
        Plot words in 2D without label
        '''
        self.pointsspacy.plot.scatter("x", "y", s=10, figsize=(20, 12))

    def plot_points_with_cluster_colors(self):
        '''
        Plot words with assigned cluster without label
        '''
        self.pointscluster.plot.scatter(x='x', y='y', c='c', cmap='tab20c', s=10, figsize=(20, 12))

    def plot_region_with_labels(self, x_bounds, y_bounds):
        '''
        Plot a region of the 2D space with labels
        '''
        slice = self.pointsspacy[
            (x_bounds[0] <= self.pointsspacy.x) &
            (self.pointsspacy.x <= x_bounds[1]) & 
            (y_bounds[0] <= self.pointsspacy.y) &
            (self.pointsspacy.y <= y_bounds[1])
            ]
    
        ax = slice.plot.scatter(x='x', y='y', s=35, figsize=(10, 8))
        for _, point in slice.iterrows():
            ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

    def plot(self):
        '''
        Show plots
        '''
        plt.show()

    def print_clusters_of(self, word):
        '''
        Print whole cluster from word 'word'
        '''
        printer = [w for w in self.word_ids if self.clusters[self.word_ids[w]] == self.clusters[self.word_ids[word]]]
        print(printer)

    def merge_cluster_values(self):
        '''
        Generate merg_dicc attribute, a dictionary where key - value is cluster - word_in_cluster
        '''
        merge_dicc = {}

        for i in range(len(self.clusters)):
            if self.clusters[i] in merge_dicc:
                value = merge_dicc[self.clusters[i]]
                value.append(list(self.word_ids.keys())[list(self.word_ids.values()).index(i)])
                merge_dicc[self.clusters[i]] = value
            else:
                merge_dicc[self.clusters[i]] = [list(self.word_ids.keys())[list(self.word_ids.values()).index(i)]]

        self.merge_dicc = merge_dicc
    
    def save_clusters(self, path_to_save, mode = 'w'):
        '''
        Save clusters in 'path_to_save'
        '''
        self.merge_cluster_values()

        i = 0
        with open(path_to_save, mode) as file:
            for index in self.merge_dicc:
                file.writelines(["Cluster ", str(i), ":\n"])
                for word in self.merge_dicc[index]:
                    file.writelines([word, ", "])
                file.write('\n\n')
                i += 1

    def apply_Word2Vec(self):
        """
        Generate Word2Vec matrix attribute representation of words
        """
        sents_tokens = []
        sents = []
        for sent in self.sents:
            for token in sent:
                if token.is_alpha and not token.is_stop:
                    sents_tokens.append(token.text)
            sents.append(sents_tokens)
            sents_tokens = []

        model = Word2Vec(sents, min_count = 1, window = 5)
        vocab = model.wv.key_to_index

        vectors = []
        word_ids = {}
        wid = 0
        for word in vocab:
            word_ids[word] = wid
            vectors.append(model.wv[word])
            wid += 1

        self.w2v_matrix = np.array(vectors)
        self.word_ids = word_ids

    def optimal_k(self):
        '''
        Find best k that maximazes the silhouette score
        '''
        print("Finding optimal k number of clusters:")
        sil = []
        matrix = self.lsa_matrix
        i = 0
        for k in range(2, 50):
            print("         -Clustering with", k, "clusters")
            km = KMeans(n_clusters = k).fit(matrix)
            labels = km.labels_
            sil.append(silhouette_score(matrix, labels, metric='euclidean'))

        
        plt.plot([k for k in range(2, 50)], sil)
        plt.show()

            

