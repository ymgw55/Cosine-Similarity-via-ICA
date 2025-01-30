"""
 Classes and function for answering analogy questions
"""

import logging
from collections import OrderedDict
from itertools import product

import pandas as pd
import scipy
import six
from six.moves import range

logger = logging.getLogger(__name__)
import sklearn
from tqdm import tqdm
from web.embedding import Embedding

from .datasets.analogy import *
from .utils import batched


class SimpleAnalogySolver(sklearn.base.BaseEstimator):
    """
    Answer analogy questions

    Parameters
    ----------
    w : Embedding instance

    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings" O. Levy et al. 2014.

    batch_size : int
      Batch size to use while computing accuracy. This is because of extensive memory usage.

    k: int
      If not None will select k top most frequent words from embedding before doing analogy prediction
      (this can offer significant speedups)

    Note
    ----
    It is suggested to normalize and standardize embedding before passing it to SimpleAnalogySolver.
    To speed up code consider installing OpenBLAS and setting OMP_NUM_THREADS.
    """

    def __init__(self, w, method="add", batch_size=300, k=None):
        self.w = w
        self.p = w.p
        self.batch_size = batch_size
        self.method = method
        self.k = k

    def score(self, X, y):
        """
        Calculate accuracy on analogy questions dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        y : array-like, shape (n_samples, )
          Analogy answers.

        Returns
        -------
        acc : float
          Accuracy
        """
        return np.mean(y == self.predict(X))

    def predict(self, X):
        """
        Answer analogy questions

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        Returns
        -------
        y_pred : array-like, shape (n_samples, )
          Predicted words.
        """
        w = self.w.most_frequent(self.k) if self.k else self.w
        p = self.p
        words = self.w.vocabulary.words
        word_id = self.w.vocabulary.word_id
        mean_vector = np.mean(w.vectors, axis=0)
        output = []

        missing_words = 0
        for query in X:
            for query_word in query:
                if query_word not in word_id:
                    missing_words += 1
        if missing_words > 0:
            logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))

        # Batch due to memory constaints (in dot operation)
        for id_batch, batch in enumerate(batched(range(len(X)), self.batch_size)):
            ids = list(batch)
            X_b = X[ids]
            if id_batch % np.floor(len(X) / (10. * self.batch_size)) == 0:
                logger.info("Processing {}/{} batch".format(int(np.ceil(ids[1] / float(self.batch_size))),
                                                            int(np.ceil(X.shape[0] / float(self.batch_size)))))

            A, B, C = np.vstack(w.get(word, mean_vector) for word in X_b[:, 0]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 1]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 2])

            if self.method == "add":
                # D = np.dot(w.vectors / np.linalg.norm(w.vectors, axis=1, keepdims=True),
                #            ((B - A + C) / np.linalg.norm(B - A + C, axis=1, keepdims=True)).T)
                # print(D.shape)
                normed_vectors = w.vectors / np.linalg.norm(w.vectors, axis=1, keepdims=True)  # shape: (vocab_size, 300)
                normed_BAC = (B - A + C) / np.linalg.norm(B - A + C, axis=1, keepdims=True)  # shape: (batch_size, 300)
                D = []
                for query in tqdm(normed_BAC):
                    # query と各単語ベクトルのアダマール積
                    hadamard_products = normed_vectors * query
                    # argsort を使い各行から top p 成分のインデックスを取得し、それらを抽出
                    top_p_indices = np.argsort(-hadamard_products, axis=1)[:, :p]

                    top_p_components = np.take_along_axis(hadamard_products, top_p_indices, axis=1)

                    # 選んだ成分の総和を求め、各クエリに対するスコアを計算
                    scores = np.sum(top_p_components, axis=1)
                    # if len(D) == 0:
                    #     print(f"hadamard_products.shape: {hadamard_products.shape}")
                    #     print(f"top_p_indices.shape: {top_p_indices.shape}")
                    #     print(f"top_p_components.shape: {top_p_components.shape}")
                    #     print(f"scores.shape: {scores.shape}")

                    D.append(scores)
                D = np.array(D).T

            elif self.method == "mul":
                assert False, "mul not supported"
                D_A = np.log((1.0 + np.dot(w.vectors, A.T)) / 2.0 + 1e-5)
                D_B = np.log((1.0 + np.dot(w.vectors, B.T)) / 2.0 + 1e-5)
                D_C = np.log((1.0 + np.dot(w.vectors, C.T)) / 2.0 + 1e-5)
                D = D_B - D_A + D_C
            else:
                raise RuntimeError("Unrecognized method parameter")

            # Remove words that were originally in the query
            for id, row in enumerate(X_b):
                D[[w.vocabulary.word_id[r] for r in row if r in
                   w.vocabulary.word_id], id] = np.finfo(np.float32).min

            output.append([words[id] for id in D.argmax(axis=0)])

        return np.array([item for sublist in output for item in sublist])


class PsAnalogySolver(sklearn.base.BaseEstimator):
    """
    Answer analogy questions

    Parameters
    ----------
    w : Embedding instance

    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings" O. Levy et al. 2014.

    batch_size : int
      Batch size to use while computing accuracy. This is because of extensive memory usage.

    k: int
      If not None will select k top most frequent words from embedding before doing analogy prediction
      (this can offer significant speedups)

    Note
    ----
    It is suggested to normalize and standardize embedding before passing it to SimpleAnalogySolver.
    To speed up code consider installing OpenBLAS and setting OMP_NUM_THREADS.
    """

    def __init__(self, w, method="add", batch_size=300, k=None):
        self.w = w
        self.ps = w.ps
        self.batch_size = batch_size
        self.method = method
        self.k = k

    def score(self, X, y):
        """
        Calculate accuracy on analogy questions dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        y : array-like, shape (n_samples, )
          Analogy answers.

        Returns
        -------
        acc : float
          Accuracy
        """
        return np.mean(y == self.predict(X))

    def predict(self, X):
        """
        Answer analogy questions

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        Returns
        -------
        y_pred : array-like, shape (n_samples, )
          Predicted words.
        """
        w = self.w.most_frequent(self.k) if self.k else self.w
        ps = self.ps
        words = self.w.vocabulary.words
        word_id = self.w.vocabulary.word_id
        mean_vector = np.mean(w.vectors, axis=0)
        p2output = dict()
        for p in ps:
            p2output[p] = []

        missing_words = 0
        for query in X:
            for query_word in query:
                if query_word not in word_id:
                    missing_words += 1
        if missing_words > 0:
            logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))

        print("sample size: {}".format(len(X)))

        # Batch due to memory constaints (in dot operation)
        for id_batch, batch in enumerate(batched(range(len(X)), self.batch_size)):
            ids = list(batch)
            X_b = X[ids]
            if id_batch % np.floor(len(X) / (10. * self.batch_size)) == 0:
                logger.info("Processing {}/{} batch".format(int(np.ceil(ids[1] / float(self.batch_size))),
                                                            int(np.ceil(X.shape[0] / float(self.batch_size)))))

            A, B, C = np.vstack(w.get(word, mean_vector) for word in X_b[:, 0]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 1]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 2])

            if self.method == "add":
                # D = np.dot(w.vectors / np.linalg.norm(w.vectors, axis=1, keepdims=True),
                #            ((B - A + C) / np.linalg.norm(B - A + C, axis=1, keepdims=True)).T)
                # print(D.shape)
                normed_vectors = w.vectors / np.linalg.norm(w.vectors, axis=1, keepdims=True)  # shape: (vocab_size, 300)
                normed_BAC = (B - A + C) / np.linalg.norm(B - A + C, axis=1, keepdims=True)  # shape: (batch_size, 300)
                p2D = dict()
                for p in ps:
                    p2D[p] = []
                    
                for query in tqdm(normed_BAC):
                    # query と各単語ベクトルのアダマール積
                    hadamard_products = normed_vectors * query
                    # argsort を使い各行から top p 成分のインデックスを取得し、それらを抽出
                    argsort_indices = np.argsort(-hadamard_products, axis=1)
                    for p in ps:
                        top_p_indices = argsort_indices[:, :p]
                        top_p_components = np.take_along_axis(hadamard_products, top_p_indices, axis=1)
                        scores = np.sum(top_p_components, axis=1)
                        p2D[p].append(scores)
                for p in ps:
                    D = p2D[p]
                    D = np.array(D).T
                    p2D[p] = D

            # Remove words that were originally in the query
            for p in ps:
                D = p2D[p]
                for id, row in enumerate(X_b):
                    D[[w.vocabulary.word_id[r] for r in row if r in
                      w.vocabulary.word_id], id] = np.finfo(np.float32).min
                p2output[p].append([words[id] for id in D.argmax(axis=0)])

        p2y_pred = dict()
        for p in ps:
            output = p2output[p]
            y_pred = np.array([item for sublist in output for item in sublist])
            p2y_pred[p] = y_pred

        return p2y_pred


class MRRAnalogySolver(sklearn.base.BaseEstimator):
    """
    Answer analogy questions

    Parameters
    ----------
    w : Embedding instance

    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings" O. Levy et al. 2014.

    batch_size : int
      Batch size to use while computing accuracy. This is because of extensive memory usage.

    k: int
      If not None will select k top most frequent words from embedding before doing analogy prediction
      (this can offer significant speedups)

    Note
    ----
    It is suggested to normalize and standardize embedding before passing it to SimpleAnalogySolver.
    To speed up code consider installing OpenBLAS and setting OMP_NUM_THREADS.
    """

    def __init__(self, w, method="add", batch_size=300, k=None):
        self.w = w
        self.batch_size = batch_size
        self.method = method
        self.k = k

    def score(self, X, y):
        """
        Calculate accuracy on analogy questions dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        y : array-like, shape (n_samples, )
          Analogy answers.

        Returns
        -------
        acc : float
          Accuracy
        """
        return np.mean(self.predict(X, y))

    def predict(self, X, y):
        """
        Answer analogy questions

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        Returns
        -------
        y_pred : array-like, shape (n_samples, )
          Predicted words.
        """
        _, dim = self.w.vectors.shape
        w = self.w.most_frequent(self.k) if self.k else self.w
        words = self.w.vocabulary.words
        word_id = self.w.vocabulary.word_id
        mean_vector = np.mean(w.vectors, axis=0)
        output = []

        missing_words = 0
        for query in X:
            for query_word in query:
                if query_word not in word_id:
                    missing_words += 1
        if missing_words > 0:
            logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))

        # Batch due to memory constaints (in dot operation)
        for id_batch, batch in enumerate(batched(range(len(X)), self.batch_size)):
            ids = list(batch)
            X_b = X[ids]
            y_b = y[ids]
            if id_batch % np.floor(len(X) / (10. * self.batch_size)) == 0:
                logger.info("Processing {}/{} batch".format(int(np.ceil(ids[1] / float(self.batch_size))),
                                                            int(np.ceil(X.shape[0] / float(self.batch_size)))))

            A, B, C = np.vstack(w.get(word, mean_vector) for word in X_b[:, 0]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 1]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 2])

            if self.method == "add":
                D = np.dot(w.vectors / np.linalg.norm(w.vectors, axis=1, keepdims=True),
                           ((B - A + C) / np.linalg.norm(B - A + C, axis=1, keepdims=True)).T)
            elif self.method == "mul":
                assert False, "mul not supported"
                D_A = np.log((1.0 + np.dot(w.vectors, A.T)) / 2.0 + 1e-5)
                D_B = np.log((1.0 + np.dot(w.vectors, B.T)) / 2.0 + 1e-5)
                D_C = np.log((1.0 + np.dot(w.vectors, C.T)) / 2.0 + 1e-5)
                D = D_B - D_A + D_C
            else:
                raise RuntimeError("Unrecognized method parameter")

            # Remove words that were originally in the query
            for id, row in enumerate(X_b):
                D[[w.vocabulary.word_id[r] for r in row if r in
                   w.vocabulary.word_id], id] = np.finfo(np.float32).min

            # calculate RR
            # D: (vocab_size, batch_size)
            sublist = []
            for id, r in enumerate(y_b):
                if dim == 1 or r not in w.vocabulary.word_id:
                    sublist.append(0)
                else:
                    wid = w.vocabulary.word_id[r]
                    y_cos = D[wid, id]
                    rank = np.sum(D[:, id] > y_cos) + 1
                    sublist.append(1 / rank)
            output.append(sublist)

        return np.array([item for sublist in output for item in sublist])