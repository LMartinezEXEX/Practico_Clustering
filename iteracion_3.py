import spacy
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

FILE_NAME = "Corpus/LaVanguardia.txt"

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

def get_processed_text(doc):
    '''Preprocess text removeing sents with less than 10 words
    and creating a dict with all words in the sentences'''
    sents = [sent for sent in doc.sents if len(sent) > 10]

    text = []
    dicc = {}
    for sent in sents:
        for token in sent:
            if token.is_alpha and len(token.text) > 1:
                text.append(token.lemma_)
                if not token.lemma_ in dicc:
                    dicc[token.lemma_] = 0
                dicc[token.lemma_] += 1

    return [sent.text for sent in sents], dicc

def TF_IDF(sents):
    '''Apply TfidfVectorizer to get matrix from sents'''

    tfidf_vectorizer = TfidfVectorizer(analyzer='word')
    tfidf = tfidf_vectorizer.fit_transform(sents)

    return tfidf_vectorizer, tfidf

def Count_Vectorizer(sents):
    '''Apply CountVectorizer to get matrix from sents'''

    c_vectorizer = CountVectorizer(analyzer='word')
    cv = c_vectorizer.fit_transform(sents)

    return c_vectorizer, cv

def LSA(matrix):
    '''Embedding to reduce dimensionality using LSA (single value descomposition)'''

    lsa = TruncatedSVD(n_components=100)
    return lsa.fit_transform(matrix)

def Kmeans(matrix):
    '''Apply K-means algorith to matrix'''

    vectors = preprocessing.normalize(matrix)
    return KMeans(n_clusters=50).fit(vectors)

def show_results(vocabulary,features,model):
    '''Print clusters with words within it and frecuent words'''

    c = Counter(sorted(model.labels_))
    print("\nTotal clusters:",len(c))
    for cluster in c:
	    print ("Cluster#",cluster," - Total words:",c[cluster])

	# Show top terms and words per cluster
    print("Top terms and words per cluster:")
    print()
	#sort cluster centers by proximity to centroid
    order_centroids = model.cluster_centers_.argsort()[:, ::-1] 

    keysFeatures = list(features.keys())
    keysVocab = list(vocabulary.keys())
    for n in range(len(c)):
        print("Cluster %d" % n)
        print("Frequent terms:", end='')
        for ind in order_centroids[n, :10]:
            print(' %s' % keysFeatures[ind], end=',')
        
        print()
        print("Words:", end='')
        word_indexs = [i for i,x in enumerate(list(model.labels_)) if x == n]
        for i in word_indexs:
            print(' %s' % keysVocab[i], end=',')
        print()
        print()

    print()

def save_clusters(title, mode, vocabulary, features, model):
    '''Save clusters to file'''

    c = Counter(sorted(model.labels_))
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    keysFeatures = list(features.keys())
    keysVocab = list(vocabulary.keys())

    with open('Resultados/iteracion_3_CLUSTERS.txt', mode) as file:
        file.write("~~~ " + title + " ~~~\n\n")
        file.writelines(["\nTotal clusters:", str(len(c)), '\n'])
        for cluster in c:
            file.writelines(["Cluster#", str(cluster)," - Total words:", str(c[cluster]), '\n'])
        
        file.write("\nTop terms and words per cluster:\n\n")

        for n in range(len(c)):
            file.writelines(["Cluster ", str(n), '\n'])
            terms = []
            for ind in order_centroids[n, :10]:
                terms.append(keysFeatures[ind] + ", ")

            terms.append('\n')
            file.writelines(["Frequent terms: "] + terms)
            
            word_indexs = [i for i,x in enumerate(list(model.labels_)) if x == n]
            words = []
            for i in word_indexs:
                words.append(keysVocab[i] + ", ")
            
            words.append('\n\n')
            file.writelines(["Words: "] + words)
            file.write('\n')
        
        file.write('\n\n')

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
    file_content = get_dataset(600_000)

    doc = spacy_processing(file_content)

    sents, vocab = get_processed_text(doc)

    tfidf_v, tfidf = TF_IDF(sents)

    c_v, cv = Count_Vectorizer(sents)

    tfidf = LSA(tfidf)

    cv = LSA(cv)

    #optimal_k(tfidf)

    #optimal_k(cv)

    tfidf_k_model = Kmeans(tfidf)
    
    cv_k_model = Kmeans(cv)

    print("~~~ TF-IDF ~~~")
    show_results(vocab, tfidf_v.vocabulary_, tfidf_k_model)

    print("~~~ Count Vectorizer ~~~")
    show_results(vocab, c_v.vocabulary_, cv_k_model)

    save_clusters("TF-IDF", 'w', vocab, tfidf_v.vocabulary_, tfidf_k_model)
    save_clusters("Count Vectorizer", 'a', vocab, c_v.vocabulary_, cv_k_model)

    
