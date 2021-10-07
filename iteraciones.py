from Word_Clustering import WordClustering

CORPUS_PATH = "Corpus/LaVanguardia.txt"

def iteracion_1(path, size):
    wc = WordClustering(path, size)
    wc.get_dataset(clean=False)
    wc.spacy_pipe_processing()
    wc.get_tokens()
    wc.get_token_frequency(min_frequency=10)
    wc.vector_generator(min_frequency=20)
    wc.generate_feature_matrix()
    wc.apply_TSNE()
    wc.apply_LSA()
    wc.apply_Kmeans_clustering("LSA", 20)
    wc.save_clusters("Resultados/iteracion_1_CLUSTERS.txt")

    del wc

def iteracion_2(path, size):
    wc = WordClustering(path, size)
    wc.get_dataset()
    wc.spacy_pipe_processing()
    wc.get_tokens()
    wc.get_token_frequency(min_frequency=10)
    wc.vector_generator(features_to_use=["WIN"], window=(0, 1), min_frequency=30)
    wc.generate_feature_matrix(process=False)
    wc.apply_TSNE()
    wc.apply_LSA()
    wc.apply_Kmeans_clustering("LSA", 40)
    wc.save_clusters("Resultados/iteracion_2_CLUSTERS.txt")

    del wc

def iteracion_3(path, size):
    wc = WordClustering(path, size)
    wc.get_dataset()
    wc.spacy_pipe_processing()
    wc.get_sents(min_frequency=10)
    wc.get_tokens(source="SENTS")
    wc.apply_Word2Vec()
    wc.apply_TSNE("W2V")
    wc.apply_Kmeans_clustering("W2V", 50)
    wc.save_clusters("Resultados/iteracion_3_CLUSTERS.txt")

    del wc

if __name__ == "__main__":
    iteracion_1(CORPUS_PATH, 200_000)

    iteracion_2(CORPUS_PATH, 300_000)

    iteracion_3(CORPUS_PATH, 300_000)