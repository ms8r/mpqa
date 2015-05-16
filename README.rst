Feature Set based on MPQA Corpus
================================

``mpqa_features.pickle`` is a serialized Pandas DataFrame with about 8000 labeled feature vectors that were derived from the annotated `MPQA Corpus`_ to model phrase and sentence level sentiment (polarity) in news and editorial content.


Basic Idea
----------

The idea is to use "subjectivity clues" - i.e. words that typically indicate expression of subjective opinion or judgement - that have been tagged with a prior polarity (i.e. the sentiment they convey in the absence of any context - neutral, positive, negative or both) to identify text passages that are likely to contain subjective expression. The subjectivity clues were taken from the `MPQA Subjectivity Lexicon`_.

The subjectivity clues were looked for in the approx. 700 documents in the extended `MPQA Corpus`_. The corpus annotations were used to determine the contextual polarity for each of occurrence. See `Wilson, Wiebe, and Hoffmann`_ (2005).


Data Set
--------

The DataFrame in ``mpqa_features.pickle`` has the following structure:


Index
~~~~~

The index consists of tuples of the form ``(path, document, count)`` where ``path`` and ``document`` identify the corpus document from which the record was generated and ``count`` is a running number within the document.

Columns
~~~~~~~

``word``: int
    The subjectivity clue (id in the `spaCy`_ vocabulary)
``word_``: string
    The subjectivity clue (i.e. an occurrence of one of the entries in the `MPQA Subjectivity Lexicon`_)
``pos``: int
    Part-of-speech (id as assigned by `spaCy`_ as a ``token``'s ``pos`` attribute)
``pos_``: string
    Part-of-speech tag (abbreviation as assigned by `spaCy`_ as a ``token``'s ``pos_``  attribute)
``before``: int
    Word preceding the subjectivity clue (vocabulary id)
``before_``: string
    Word preceding the subjectivity clue
``after``: int
    Word following the subjectivity clue (vocabulary id)
``after_``: string
    Word following the subjectivity clue
``context``: tuple of ints
    (before, word, after) as vocabulary ids (this is redundant with the individual tokens but helps for visually exploring the data set).
``context_``: tuple if strings
    (before, word, after)
``pre_neg``: boolean
    True if there is a negating word within four tokens ahead of the subjectivity clue (not counting double negations such as "not only")
``post_neg``: boolean
    True if there is a negating word within four tokens after of the subjectivity clue
``pri_pol``: string
    Prior polarity as given in the `MPQA Subjectivity Lexicon`_ (possible values are 'positive', 'negative', 'both', and 'neutral')
``rel``: string
    Reliability class as given in the `MPQA Subjectivity Lexicon`_ (possible values are 'strongsubj', 'weaksubj')
``c_pol``: float
    Contextual polarity, derived from the contextual polarity and intensity entries in the `MPQA Corpus`_: uses 1.0 for positive and -1.0 for neagtive polarity (0 for neutral) and multiplies with the intensity annotation (0 for low, 0.75 for medium, 1.5 for high, and 2.0 for extreme).
``is_int``: boolean
    True if the subjectivity clue itself is an intensifier. The list of itensifiers is taken from the `MPQA Arguing Lexicon`_.
``prec_int``: boolean
   True if subjectivity clue is preceded by an intensifier
``prec_adj``: boolean
   True if subjectivity clue is preceded by an adjective
``prec_adv``: boolean
   True if subjectivity clue is preceded by an adverb
``topic``: string
   Topic of the respective article (if available in the corpus annotations)
``pword``: int
   "Packed id" for ``word`` (the subjectivity clue). Between the ``word``, ``before`` and ``after`` vocabulary ids there are 3,691 different ids, taken from a vocabulary with about 300,000 entries. The "packed" values map these ids onto consecutive integers between 0 and 3,690.
``pbefore``: int
    Packed id for ``before``
``pbafter``: int
    Packed id for ``after``


Python Script
-------------

The python script ``mpqa.py`` was used to construct the labeled feature vectors in ``mpqa_features.pickle``. Run::

    python mpqa mkfeat -h

for instructions. Running the script requires (see `Licenses for Corpus Content and Annotations`_):

* Third party python packages `pandas`_ and `spaCy`_.

* The annotated `MPQA Corpus`_.

* A 'doclist' file that lists all documents to be included in the data set  (concatenate and dedupe the partially overlapping doclists that come with MPQA Corpus or use the ``doclist.combinedUnique`` file in this repo).

* The `MPQA Subjectivity Lexicon`_ (provided as ``subjclues.tff`` in this repo).

* A list of intensifiers, available in the `MPQA Arguing Lexicon`_ (file ``intensifiers.tff``).

To use the ``mpqa`` module from within your own script follow this example::

    from __future__ import print_function
    import pandas as pd
    import mpqa

    df = pd.DataFrame(columns=mpqa.FEAT_COLS)
    for path, fname, topic in mpqa.iter_docs('doclist.combinedUnique'):
        print(path, fname)
        doc = mpqa.Doc(
                mpqa_dir='database.mpqa.2.0',
                path=path,
                fname=fname,
                topic=topic,
                sc_path='subjclues.tff',
                int_path='intensifiers.tff')
        df = df.append(doc.feat_df)

    sparse_cols = ['word', 'before', 'after']
    pack_cols = mpqa.pack_df(df, sparse_cols)
    for c in sparse_cols:
        df['p' + c] = pack_cols[c]

This assumes that you have downloaded and extracted the `MPQA Corpus`_ to ``database.mpqa.2.0``. The resulting DataFrame ``df`` will be the same as the one that can be obtained by unpickling ``mpqa_features.pickle``.


Licenses for Corpus Content and Annotations
-------------------------------------------

The `download site`_ for the `MPQA Corpus`_ and annotations states the following licensing terms:

    The annotations in this data collection are copyrighted by the MITRE Corporation. User acknowledges and agrees that: (i) as between User and MITRE, MITRE owns all the right, title and interest in the Annotated Content, unless expressly stated otherwise; (ii) nothing in this Agreement shall confer in User any right of ownership in the Annotated Content; and (iii) User is granted a non-exclusive, royalty free, worldwide license (with no right to sublicense) to use the Annotated Content solely for academic and research purposes. This Agreement is governed by the law of the Commonwealth of Massachusetts and User agrees to submit to the exclusive jurisdiction of the Massachusetts courts.

    Note: The textual news documents annotated in this corpus have been collected from a wide range of sources and are not copyrighted by the MITRE Corporation. The user acknowledges that the use of these news documents is restricted to research and/or academic purposes only.

The `MPQA Subjectivity Lexicon`_ and the `MPQA Arguing Lexicon`_ are provided under a GNU General Public License.


.. _MPQA Corpus: http://mpqa.cs.pitt.edu/
.. _MPQA Subjectivity Lexicon: http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/
.. _spaCy: https://honnibal.github.io/spaCy/index.html
.. _MPQA Arguing Lexicon: http://mpqa.cs.pitt.edu/lexicons/arg_lexicon/
.. _Wilson, Wiebe, and Hoffmann: http://www.cs.pitt.edu/~wiebe/pubs/papers/emnlp05polarity.pdf
.. _download site: http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/
.. _pandas: https://pypi.python.org/pypi/pandas/0.15.2
