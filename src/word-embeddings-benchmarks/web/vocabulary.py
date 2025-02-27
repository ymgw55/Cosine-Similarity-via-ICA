#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Supports word embeddings.

NOTE: This file was adapted from the polyglot package
"""

import os
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from io import StringIO, open

import six
from six import iteritems, string_types
from six import text_type as unicode
from six.moves import zip

from .utils import _open


def count(lines):
    """ Counts the word frequences in a list of sentences.

    Note:
      This is a helper function for parallel execution of `Vocabulary.from_text`
      method.
    """
    words = [w for l in lines for w in l.strip().split()]
    return Counter(words)


class Vocabulary(object):
    """ A set of words/tokens that have consistent IDs.

    Attributes:
      word_id (dictionary): Mapping from words to IDs.
      id_word (dictionary): A reverse map of `word_id`.
    """

    def __init__(self, words=None):
        """ Build attributes word_id and id_word from input.

        Args:
          words (list/set): list or set of words.
        """
        words = self.sanitize_words(words)
        self.word_id = {w: i for i, w in enumerate(words)}
        self.id_word = {i: w for w, i in iteritems(self.word_id)}

    def __iter__(self):
        """Iterate over the words in a vocabulary."""
        for w, i in sorted(iteritems(self.word_id), key=lambda wc: wc[1]):
            yield w

    @property
    def words(self):
        """ Ordered list of words according to their IDs."""
        return list(self)

    def __unicode__(self):
        return u"\n".join(self.words)

    def __str__(self):
        if six.PY3:
            return self.__unicode__()
        return self.__unicode__().encode("utf-8")

    def __getitem__(self, key):
        if isinstance(key, string_types) and not isinstance(key, unicode):
            key = unicode(key, encoding="utf-8")
        return self.word_id[key]

    def add(self, word):
        if isinstance(word, string_types) and not isinstance(word, unicode):
            word = unicode(word, encoding="utf-8")

        if word in self.word_id:
            raise RuntimeError("Already existing word")

        id = len(self.word_id)
        self.word_id[word] = id
        self.id_word[id] = word

    def __contains__(self, key):
        return key in self.word_id

    def __delitem__(self, key):
        """Delete a word from vocabulary.

        Note:
         To maintain consecutive IDs, this operation implemented
         with a complexity of \\theta(n).
        """
        del self.word_id[key]
        self.id_word = dict(enumerate(self.words))
        self.word_id = {w: i for i, w in iteritems(self.id_word)}

    def __len__(self):
        return len(self.word_id)

    def sanitize_words(self, words):
        """Guarantees that all textual symbols are unicode.
        Note:
          We do not convert numbers, only strings to unicode.
          We assume that the strings are encoded in utf-8.
        """
        _words = []
        for w in words:
            if isinstance(w, string_types) and not isinstance(w, unicode):
                _words.append(unicode(w, encoding="utf-8"))
            else:
                _words.append(w)
        return _words

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError as e:
            return default

    def getstate(self):
        return list(self.words)

    @classmethod
    def from_vocabfile(cls, filename):
        """ Construct a CountedVocabulary out of a vocabulary file.

        Note:
          File has the following format word1
                                        word2
        """
        words = [x.strip() for x in _open(filename, 'r').read().splitlines()]
        return cls(words=words)


class OrderedVocabulary(Vocabulary):
    """ An ordered list of words/tokens according to their frequency.

    Note:
      The words order is assumed to be sorted according to the word frequency.
      Most frequent words appear first in the list.

    Attributes:
      word_id (dictionary): Mapping from words to IDs.
      id_word (dictionary): A reverse map of `word_id`.
    """

    def __init__(self, words=None):
        """ Build attributes word_id and id_word from input.

        Args:
          words (list): list of sorted words according to frequency.
        """

        words = self.sanitize_words(words)
        self.word_id = {w: i for i, w in enumerate(words)}
        self.id_word = {i: w for w, i in iteritems(self.word_id)}

    def most_frequent(self, k):
        """ Returns a vocabulary with the most frequent `k` words.

        Args:
          k (integer): specifies the top k most frequent words to be returned.
        """
        return OrderedVocabulary(words=self.words[:k])


class CountedVocabulary(OrderedVocabulary):
    """ List of words and counts sorted according to word count.
    """

    def __init__(self, word_count=None):
        """ Build attributes word_id and id_word from input.

        Args:
          word_count (dictionary): A dictionary of the type word:count or
                                   list of tuples of the type (word, count).
        """

        if isinstance(word_count, dict):
            word_count = iteritems(word_count)
        sorted_counts = list(sorted(word_count, key=lambda wc: wc[1], reverse=True))
        words = [w for w, c in sorted_counts]
        super(CountedVocabulary, self).__init__(words=words)
        self.word_count = OrderedDict(sorted_counts)

    def most_frequent(self, k):
        """ Returns a vocabulary with the most frequent `k` words.

        Args:
          k (integer): specifies the top k most frequent words to be returned.
        """
        word_count = [(w, self.word_count[w]) for w in self.words[:k]]
        return CountedVocabulary(word_count=word_count)

    def min_count(self, n=1):
        """ Returns a vocabulary after eliminating the words that appear < `n`.

        Args:
          n (integer): specifies the minimum word frequency allowed.
        """
        word_count = [(w, c) for w, c in iteritems(self.word_count) if c >= n]
        return CountedVocabulary(word_count=word_count)

    def __unicode__(self):
        return u"\n".join([u"{}\t{}".format(w, self.word_count[w]) for w in self.words])

    def __delitem__(self, key):
        super(CountedVocabulary, self).__delitem__(key)
        self.word_count = OrderedDict([(w, self.word_count[w]) for w in self])

    def getstate(self):
        words = list(self.words)
        counts = [self.word_count[w] for w in words]
        return (words, counts)

    @staticmethod
    def from_vocabfile(filename):
        """ Construct a CountedVocabulary out of a vocabulary file.

        Note:
          File has the following format word1 count1
                                        word2 count2
        """
        word_count = [x.strip().split() for x in _open(filename, 'r').read().splitlines()]
        word_count = OrderedDict([(w, int(c)) for w, c in word_count])
        return CountedVocabulary(word_count=word_count)


