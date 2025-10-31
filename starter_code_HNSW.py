import faiss
import h5py
import numpy as np
import os
import requests


def read_fvecs(filename):
    data = np.fromfile(filename, dtype='int32')
    d = data[0]
    data = data.reshape(-1, d + 1)
    data = data[:, 1:].astype('float32')
    return data


def evaluate_hnsw():
    os.system("wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 -O sift.h5")

    with h5py.File("sift.h5", "r") as f:
        base_embeddings = f["train"][:]   # Database vectors
        query_embeddings = f["test"][:]   # Query vectors

    print("Base embeddings shape:", base_embeddings.shape)
    print("Query embeddings shape:", query_embeddings.shape)

    d = base_embeddings.shape[1]  # dimensionality of vectors
    index = faiss.IndexHNSWFlat(d, 16)  # M = 16
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 200

    index.add(base_embeddings.astype('float32'))
    print("Total vectors in index:", index.ntotal)
    query_vector = query_embeddings[0:1].astype('float32')  # shape (1, d)
    D, I = index.search(query_vector, 10)  # top 10 nearest neighbors
    with open("./output.txt", "w") as f:
        for idx in I[0]:
            f.write(f"{idx}\n")

    print("Top 10 nearest neighbor indices written to output.txt")
    return 0

if __name__ == "__main__":
    evaluate_hnsw()
