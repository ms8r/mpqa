#!/usr/bin/env python

"""
Processes MPQA corpus
"""

from __future__ import division, print_function, unicode_literals
import os
from collections import namedtuple, OrderedDict
from itertools import islice
import json
import numpy as np
import pandas as pd
import spacy.en
import argparse
import logging


PRI_POL_MAP = {
        'negative': -1,
        'neutral': 0,
        'positive': 1,
        'both': 0,
        }

ColFlags = namedtuple('ColFlags', ['kind', 'transform', 'ref'])
# 'kind' can be:
#   'data'  - keep as data colum
#   'label' - results/labels for training and testing data
#   any other value will not be taken over for processing
# 'transform' can be:
#   'pack'  -   pack to consequtive ints ('ref' provides map to use)
#               packed cols will be converted to (0,1) arrays
#   'raw'   -   keep as is
#   'map'   -   map values using the map provided under 'ref'
# 'ref':
#   Callable to be called to transform 'pack' and 'map' columns. Must accept
#   one argument. For'map' this will be called for straight element wise
#   mapping. For 'pack' it is a callalble which as its sole argument will
#   receive a numpy array with all columns flagged with this callable and which
#   return a dict which can be used to map each of these columns.

def pos_map(k):
    pack_map(k)


def voc_map(k):
    pack_map(k)


COL_MAP = OrderedDict([
    ('word', ColFlags('data', 'pack', voc_map)),
    # the subj. clue word (id)
    ('word_', ColFlags('', '', None)),
    # the subj. clue word (string)
    ('pos', ColFlags('data', 'pack', pos_map)),
    # part-of-speach (id)
    ('pos_', ColFlags('', '', None)),
    # part-of-speach (string)
    ('before', ColFlags('data', 'pack', voc_map)),
    # token preceding word (id)
    ('before_', ColFlags('', '', None)),
    # token preceding word (string)
    ('after', ColFlags('data', 'pack', voc_map)),
    # token following word (id)
    ('after_', ColFlags('', '', None)),
    # token following word (string)
    ('context',  ColFlags('', '', None)),
    # tuple (before, word, after)
    ('context_',  ColFlags('', '', None)),
    # tuple (before, word, after) with strings
    ('pre_neg',  ColFlags('data', 'raw', None)),
    # negation in front of token
    ('post_neg',  ColFlags('data', 'raw', None)),
    # negation after token
    ('pri_pol',  ColFlags('data', 'map', lambda k: PRI_POL_MAP[k])),
    # prior polarity
    ('rel',  ColFlags('data', 'map', lambda k: 1 if k == 'strongsubj' else 0)),
    # reliability class {strongsubj, weaksubj}
    ('c_pol',  ColFlags('label', 'raw', None)),
    # contextual polarity
    ('is_int',  ColFlags('data', 'raw', None)),
    # is intensifier (boolean)
    ('prec_int',  ColFlags('data', 'raw', None)),
    # preceded by intensifier (boolean)
    ('prec_adj',  ColFlags('data', 'raw', None)),
    # preceded by adjective (boolean)
    ('prec_adv',  ColFlags('data', 'raw', None)),
    # preceded by adverb (boolean)
    ('topic',  ColFlags('', '', None)),
    # topic (if available)
])

FEAT_COLS = COL_MAP.keys()

Annot = namedtuple('Annot', ['idnum', 'start', 'end', 'ref',
                             'kind', 'gate', 'attr'])
GateSent = namedtuple('GateSent', ['start', 'end'])
SubClue = namedtuple('SubClue', ['rel', 'pri_pol', 'stemmed'])
Polar = namedtuple('Polar', ['token', 'rel', 'pri_pol', 'c_pol'])

logging.basicConfig(level=logging.DEBUG)


class Intensifier(object):
    """
    Builds set of (word, POS) tuples from intensifier file. Provides method to
    match (word, POS) tuple against itself.
    """

    _int_pos_map = {'@INTENSADV1': 'ADV', '@INTENSADJ1': 'ADJ'}

    def __init__(self, int_path='arglex_Somasundaran07/intensifiers.tff'):
        # load intensifiers:
        with open(int_path, 'r') as fo:
            lines = [tuple(s.strip().split('=')) for s in fo
                     if not s.startswith('#')]
        intensifier = set()
        for pos, words in lines:
            i_pos = self._int_pos_map[pos]
            for w in words.strip('{}').split(', '):
                intensifier.add((w, i_pos))

        self.lex = intensifier

    def lookup(self, token):
        if ((token.norm_, token.pos_)) in self.lex:
            return True
        else:
            return False


class SubjectivityClues(object):
    """
    Subjectivity clues. Builds a dict that maps (word, pos) tuples to
    SubClue named tuples from an input file in the format referenced in
    'Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis'
    (Wilson, Wiebe, Hoffmann, 2005).  Provides methods to match (word, POS)
    pairs against the lexicon and return prior polarity and reliability.
    """
    # map POS tags in subjectivty clues to spacy tags:
    _pos_map = {
        'noun': 'NOUN',
        'verb': 'VERB',
        'adj': 'ADJ',
        'adverb': 'ADV',
        'anypos': 'ANYPOS',
        }

    def __init__(self, sc_path=
            'subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'):

        # load subjectivity clues
        with open(sc_path, 'r') as fo:
            sub_clues = [c.strip().replace('type=', 'rel=').split()
                         for c in fo]
        sub_clues = [dict([d.split('=') for d in sc if len(d.split('=')) == 2])
                     for sc in sub_clues]

        # convert to dict keyed by (word, part-of-speech) and exclude neutral
        # polarity clues:
        # ISSUE: if stemmed for matches non-stemmed will overwrite (may not
        # have impact as polarity and type remain the same
        self.lex = {(s['word1'], self._pos_map[s['pos1']]): SubClue(s['rel'],
                     s['priorpolarity'],
                     True if s['stemmed1'] == 'y' else False)
                    for s in sub_clues if s['priorpolarity']}
        # sclues_pos and sclues_any are just sets containing all (word, pos)
        # tuples with a proper POS tag and 'ANYPOS' respectively
        self.sclues_pos = set([(w, t) for w, t in self.lex.keys()
                               if t != 'ANYPOS'])
        self.sclues_any = set(self.lex.keys()).difference(self.sclues_pos)

    def lookup(self, token):
        """
        Returns a `SubClue` named tuple for `token` if either `(token.norm_,
        token.pos_)` or `(token.norm_, 'ANYPOS') is in self.lex. Returns an
        empty tuple if no match is found.
        """
        tnp = (token.norm_, token.pos_)
        tna = (token.norm_, 'ANYPOS')
        if tnp in self.lex or tna in self.lex:
            key = tnp if tnp in self.lex else tna
            return self.lex[key]
        else:
            return tuple()


class Doc(object):
    """
    Wraps all data for a document in the MPQA corpus
    - Text of the document in `tokens`, a spacy tokens object
    - List of annotations (`Annot` named tuples) in `annot`, sorted by start
      character position. We'll only keep *expressive-subjectivity* and
      *direct-subjective* annotations as these are the only ones that include
      contextual polarity markers. If the dict under `attr` has the
      keyword`'polarity'` this indicates Indicates the contextual polarity of
      the private state (what we're after). Possible values: positive,
      negative, both, neutral, uncertain-positive, uncertain-negative,
      uncertain-both, uncertain-neutral. Use `'intensity'` rather than
      `'expression-intensity'` attribute.
    - Sentence boundaries in `gate_sent` (list of `GateSent` named tuples)

    Attributes:
    ----------

        mpqa_dir: string
            Path to the MPQA root directory
        path: string
            Path to the corpus doc, relative to mpqa_dir
        fname: string
            File name of the doc
        topic: string
            Topic of the doc (if provided at initialization)
        annot: list of `Annot` named tuples
            Annotations for the doc (only keeps annotations for
            'GATE_direct-subjective' and 'GATE_expressive-subjectivity')
        gate_sent: list of `GateSent` named tuples
            Sentence boundaries in doc
        polars: list of Polar named tuples
            Prior polarity and context polarity for subjectivity clue
            occurrences in doc
        features: list of dicts
            Features contructed from `polars` and document context; key are the
            items of the global `FEAT_COLS`
        feat_df: pandas DataFrame
            `features` as a DataFrame, index by `(path, fname, num)` tuples,
            where `num` is a running number within the doc

    Note: there is a sequence `polars` -> `features` -> `feat_df` as each
    bulids on the preceding one. To save memory, `features` will assign
    `polars` back to `None` and `feat_df` will do the same with `features`. All
    three are lazily computed properties so they would just be recreated if
    referenced after their 'successor' has alreday been built.
    """
    # directory structure for corpus
    _doc_dir = 'docs'
    _ann_dir = 'man_anns'
    _ann_name = 'gateman.mpqa.lre.2.0'
    _gate_sent_name = 'gatesentences.mpqa.2.0'

    # polarity and intensity mappings:
    _pol_map = {'negative': -1, 'positive': 1}
    _int_map = {'low': 0, 'medium': 0.75, 'high': 1.5, 'extreme': 2}

    _negations = set([u'no', u'not', u'neither', u'nor', u'nobody', u'none',
                      u'nothing', u'n\'t'])
    _doub_negs = set([u'not only'])
    # number of tokens to check before and after for neagtion:
    _neg_span = 4

    _nlp = spacy.en.English()

    def __init__(self, mpqa_dir, path, fname, topic, sc_path, int_path):
        """
        Args:
            mpqa_dir: string
                MPQA root directory; all other parameters are relative to this
                directory
            path: string
                path to document, relative to `mpqa_dir`
            fname: string
                document name
            topic: string
                document topic
            sc_path: string
                path to subjectivity clues file
            int_path: string
                path to intensifier file
        """
        self.mpqa_dir = mpqa_dir
        self.path = path
        self.fname = fname
        self.topic = topic

        self._sc = SubjectivityClues(sc_path)
        self._in = Intensifier(int_path)

        # first document:
        with open(os.path.join(mpqa_dir, self._doc_dir,
                  path, fname), 'r') as fo:
            text = fo.read().decode('utf-8')
            self.tokens = self._nlp(text)

        # then annotations:
        with open(os.path.join(mpqa_dir, self._ann_dir, path, fname,
                  self._ann_name), 'r') as fo:
            annot = [a for a in fo if not a.startswith('#')]

        annot = [tuple(a.split('\t')) for a in annot]
        annot = [(int(idnum), int(b.split(',')[0]), int(b.split(',')[1]),
                  kind, gate, attr.strip().replace(', ', ',').split('" '))
                 for idnum, b, kind, gate, attr in annot]

        annot = [Annot(idnum=idnum, start=start, end=end,
                       ref=text[start:end].lower(), kind=kind, gate=gate,
                       attr={t.split('=')[0]: t.split('=')[1].strip('"')
                             for t in attr})
                 for idnum, start, end, kind, gate, attr in annot
                 if gate in ('GATE_direct-subjective',
                             'GATE_expressive-subjectivity') and attr[0]]

        self.annot = [a for a in annot if 'polarity' in a.attr]
        self.annot.sort(key=lambda k: k.start)

        # Now sentence boundaries:
        with open(os.path.join(mpqa_dir, self._ann_dir, path, fname,
                  self._gate_sent_name), 'r') as fo:
            gate_sent = [a.strip().split('\t') for a in fo
                         if not a.startswith('#')]

        gate_sent = [(int(b.split(',')[0]), int(b.split(',')[1]))
                     for _, b, _, _ in gate_sent]

        self.gate_sent = [GateSent(start=s, end=e) for s, e in gate_sent]
        self.gate_sent.sort(key=lambda k: k.start)

        self._polars = None
        self._features = None
        self._feat_df = None

    @property
    def polars(self):
        """
        Given `doc`, a `Doc` object,  a `SubClues` object, `polars` will return
        a list of (token, SubClue, contextual polarity) tuples
        """
        if self._polars is not None:
            return self._polars

        def context_pol(token):
            """
            Checks annotations for all spans that overlap `token` and returns
            a composite contextual polarity as the dot product between polarity
            and intensity.
            """
            matches = [self._pol_map.get(a.attr['polarity'], 0) *
                       self._int_map.get(a.attr.get('intensity', 'medium'), 0)
                       for a in self.annot
                       if a.start <= token.idx and a.end > token.idx]
            return sum(matches)

        # build list of (token, SubClue, contextual polarity) tuples
        sc_tokens = []
        for t in self.tokens:
            prior = self._sc.lookup(t)
            if prior:
                ctp = context_pol(t)
                if ctp:
                    sc_tokens.append(Polar(token=t, rel=prior.rel,
                            pri_pol=prior.pri_pol, c_pol=ctp))

        self._polars = sc_tokens
        return self._polars

    def _negation(self, ti, direction):
        """
        Returns True if a negating word is present around token with index
        ``ti`` with ``self._neg_span`` tokens ahead (if ``direction < 0``) or
        after (if ``direction > 0``) position ``ti``, else False. Terms in
        ``self._doub_negs`` will be exlcuded from negations.
        """
        # check for negation:
        num_toks = len(self.tokens)
        if ti == 0 or ti == num_toks - 1:
            return False
        ahead = True if direction < 0 else False
        if ahead:
            pre_start = max(0, ti - self._neg_span)
            span = [s.lower_ for s in islice(self.tokens, pre_start, ti)]
        else:
            post_end = min(num_toks, ti + self._neg_span)
            span = [s.lower_ for s in islice(self.tokens, ti, post_end)]
        bigrams = [' '.join(s) for s in zip(span, span[1:])]
        span = set(span + bigrams)
        neg = (True if span.intersection(self._negations) and not
               span.intersection(self._doub_negs) else False)

        return neg

    def _context(self, ti, max_prob=0, keep_neg=True, skip_punct=False):
        """
        Returns a (before, after) tuple of the two tokens surrounding the token
        with the index ti. Only tokens with a probability of less than
        ``max_prob`` will be considered. Unless ``keep_neg`` is ``False``
        negations will be kept independent of their probability. Whitespace
        tokens will be ignored. If ``skip_punct`` is True, punctuation will
        also be ignored. If no token is found ``None`` is returned in its
        position.
        """
        num_toks = len(self.tokens)

        def check(t):
            if ((not skip_punct or t.pos_ != u'PUNCT') and
                     (t.prob <= max_prob or
                     (t.lower_ in self._negations and keep_neg) or
                     (not skip_punct and t.pos_ == u'PUNCT')) and
                     not t.string.isspace()):
                return True
            else:
                return False

        before = None
        i = 1
        while ti - i >= 0:
            t = self.tokens[ti - i]
            if check(t):
                before = t
                break
            i += 1

        after = None
        i = 1
        while ti + i < num_toks:
            t = self.tokens[ti + i]
            if check(t):
                after = t
                break
            i += 1

        return (before, after)

    @property
    def features(self):
        """
        Contructs feature vectors as a list of dicts, keyed by the items in
        the global `FEAT_COLS`.
        """
        if self._features is not None:
            return self._features

        voc = self._nlp.vocab

        feats_list = []
        for p in self.polars:
            feats = {}
            ti = p.token.i
            feats['topic'] = self.topic
            feats['word'] = voc[p.token.lower_].id
            feats['word_'] = p.token.lower_
            feats['pos'] = p.token.pos
            feats['pos_'] = p.token.pos_
            feats['pri_pol'] = p.pri_pol
            feats['c_pol'] = p.c_pol
            feats['rel'] = p.rel
            feats['is_int'] = self._in.lookup(p.token)
            # now the more difficult stuff:
            before, after = self._context(ti, max_prob=-5., skip_punct=False,
                                          keep_neg=True)
            feats['before'] = voc[before.lower_].id if before else -1
            feats['before_'] = before.lower_ if before else ''
            feats['after'] = voc[after.lower_].id if after else -1
            feats['after_'] = after.lower_ if after else ''
            feats['context'] = (feats['before'], feats['word'], feats['after'])
            feats['context_'] = (feats['before_'], feats['word_'],
                                 feats['after_'])
            pre = self.tokens[ti - 1] if ti > 0 else None
            feats['prec_int'] = 1 if pre and self._in.lookup(pre) else 0
            feats['prec_adj'] = 1 if pre and pre.pos_ == 'ADJ' else 0
            feats['prec_adv'] = 1 if pre and pre.pos_ == 'ADV' else 0
            feats['is_int'] = 1 if self._in.lookup(p.token) else 0
            # check for negation:
            feats['pre_neg'] = 1 if self._negation(ti, -1) else 0
            feats['post_neg'] = 1 if self._negation(ti, 1) else 0

            feats_list.append(feats)

        self._features = feats_list
        # free memory:
        self._polars = None
        return self._features

    @property
    def feat_df(self):
        """
        Feature vecs as DataFrame
        """
        if self._feat_df is not None:
            return self._feat_df

        columns = FEAT_COLS
        data_dict = {}
        for c in columns:
            data = [f[c] for f in self.features]
            data_dict[c] = data

        num_recs = len(self.features)
        index = zip([self.path] * num_recs, [self.fname] * num_recs,
                    range(num_recs))
        self._feat_df = pd.DataFrame(data_dict, index=index, columns=columns)
        # free memory:
        self._features = None
        return self._feat_df


def pack_map(a):
    """
    Creates a mapping of int ids in a (ndarray) onto consequtive ints and
    returns this as a dict id: packed_id
    """
    u = np.unique(a)
    pack_map = {int(k): v for k, v in zip(u, np.arange(u.size))}

    return pack_map


def make_pack_map(df, columns):
    """
    Creates a mapping of int ids in `columns` (list) of DataFrame df onto
    consequtive ints and returns this a a dict id: packed_id
    """
    items = set()
    for c in columns:
        items = items.union({int(i) for i in df[c]})
    items = sorted(list(items))
    pack_map = {k: v for k, v in zip(items, range(len(items)))}

    return pack_map


def pack_df(df, columns):
    """
    Takes integer ids in `columns` (list) for df (pandas DataFrame) and maps
    them on a set of consequtive ints. Returns a DataFrame with `columns`.
    """
    pack_map = make_pack_map(df, columns)
    df_packed = pd.DataFrame(columns=columns)
    for c in columns:
        df_packed[c] = df[c].map(lambda k: pack_map[k])

    return df_packed


def packed_to_array(packed, feat_size, weight=1):
    """
    Converts a feature vector (array like) with integer ids into a
    2-dim numpy array whith the samples as the row (1st) index and the ids
    across the 2nd index. Element [sample_no, feat_id ] will have a value of
    ``weight``, all other values will be 0. ``size`` specifies the number of
    columns (size of feature set).
    """
    samples = np.array(packed, dtype=int)
    a = np.zeros((samples.size, feat_size))
    a[np.arange(samples.size), samples] = weight

    return np.array(a, dtype=int)


def iter_docs(doc_list_fn, topics=False):
    """
    Generator over the docs listed in `doc_list_fn`.
    If `topics` is `True` assumes input lines will have the format

        topic=[TOPIC] file=[PATH/FILE]

    else the input is simply

        [PATH/FILE]

    Yields (path, name, topic) tuples with `topic == ''` if `topics` argument
    is False.
    """
    with open(doc_list_fn) as fo:
        for line in fo:
            if topics and len(line.split()) != 2:
                continue
            if topics:
                f, t = (line.split()[1].rstrip('\n').split('=')[1],
                        line.split()[0].split('=')[1])
            else:
                f, t = (line.strip(), '')

            yield (os.path.dirname(f), os.path.basename(f), t)


def handle_mkfeat(args, **kwargs):
    """
    Creates a feature DataFrame and pickles same as file.
    """
    sc_path = args.subclues.name
    args.subclues.close()
    int_path = args.intensifiers.name
    args.intensifiers.close()
    doclist = args.doclist.name
    args.doclist.close()
    outfile = args.output.name
    args.output.close()

    # build labeled feature vecs:
    df = pd.DataFrame(columns=FEAT_COLS)
    for p, f, t in iter_docs(doclist):
        logging.info('processing %s', os.path.join(p, f))
        doc = Doc(args.mpqa, p, f, t, sc_path, int_path)
        df = df.append(doc.feat_df)

    df.to_pickle(outfile)


def handle_mkdata(args, **kwargs):
    """
    Turns a feature DataFrame into an ndarray. Will only keep columns flagged
    as 'data' in COL_MAP and transform a required. Outputs are:

        data
            The transformed numerical data (all 0,1) in a sparse DataFrame with
            samples in rows and features across the columns (pickled)
        labels
            The target vector(s) as a (dense) DataFrame. The label data is kept
            as is, i.e. without any transformations (pickled).
        maps
            Mappings used to pack features. For example, if the original data
            had a feature vector [5, 1, 9] the packed version would be
            [1, 0, 2] and the mapping {1: 0, 5: 1, 9: 2}. `maps` is a
            dictionary of mapping dicts, keyed by the `__name__` attributes of
            the `ref` elements in `COL_MAP`.
    """
    infile = args.featdf.name
    args.featdf.close()
    data_file = args.data.name
    args.data.close()
    labels_file = args.labels.name
    args.labels.close()

    logging.info('reading feature DataFrame from %s...', infile)
    df_feat = pd.read_pickle(infile)
    # re-index for alignment with DataFrames created from packed ndarrays:
    df_feat.index = np.arange(len(df_feat))

    # deal with packing: columns sharing same packing reference in COL_MAP use
    # the same mapping dict.
    logging.info('building pack maps and saving in %s', args.maps.name)
    pack_groups = set([cf.ref for cf in COL_MAP.values()
                       if cf.transform == 'pack'])
    maps = {}
    for p in pack_groups:
        cols = [c for c, cf in COL_MAP.iteritems() if cf.ref == p]
        pmap = pack_map(df_feat[cols].values)
        for c in cols:
            df_feat[c] = df_feat[c].map(lambda k: pmap[k])
        maps[p.__name__] = pmap
    json.dump(maps, args.maps, indent=2)
    args.maps.close()

    # construct feature ndarray, converting packed columns into [0, 1] arrays:
    logging.info('constructing new DataFrame...')
    xdf = pd.DataFrame(index=df_feat.index)
    for c, cf in COL_MAP.iteritems():
        logging.info('processing column "%s"', c)
        if cf.kind != 'data':
            continue
        if cf.transform == 'raw':
            logging.debug("processing as 'raw', len(xdf) is %d", len(xdf))
            xdf = pd.concat([xdf, df_feat[c].astype('int32')], axis=1)
            logging.debug("after concat, len(xdf) is %d", len(xdf))
        if cf.transform == 'map':
            xdf = pd.concat([xdf, df_feat[c].map(cf.ref)], axis=1)
        if cf.transform == 'pack':
            feat_size = len(maps[cf.ref.__name__])
            logging.debug('number of packed features: %d', feat_size)
            width = int(np.log10(feat_size)) + 1
            a = packed_to_array(df_feat[c], feat_size, weight=1)
            logging.debug('packed array shape: %s', a.shape)
            acols = ['{colname}_{index:0{width}}'.format(colname=c,
                     width=width, index=i) for i in xrange(feat_size)]
            xdf = pd.concat([xdf, pd.DataFrame(a, columns=acols)], axis=1)
        logging.debug('shape of xdf: %s', xdf.values.shape)
    xdf.to_sparse(fill_value=0).to_pickle(data_file)
    logging.info('new DataFrame pickled in %s', data_file)

    # extract and save labels:
    logging.info('extracting labels and pickling in %s', labels_file)
    label_cols = [c for c, cf in COL_MAP.iteritems() if cf.kind == 'label']
    df_feat[label_cols].to_pickle(labels_file)


def setup_parser_mkfeat(p):
    p.add_argument('--mpqa', default='.', help='path to mpqa root directory; '
                   'assumes current dir if not provided')
    p.add_argument('--subclues', metavar='FILE',
                   type=argparse.FileType('r'), required=True,
                   help='path to subjectivity clues file')
    p.add_argument('--intensifiers', metavar='FILE',
                   type=argparse.FileType('r'), required=True,
                   help='path to intensifiers file')
    p.add_argument('--doclist', metavar='FILE',
                   type=argparse.FileType('r'), required=True,
                   help='path to doclist file')
    p.add_argument('--output', metavar='FILE',
                   type=argparse.FileType('w'), required=True,
                   help='output file in which to store pickled dataframe')


def setup_parser_mkdata(p):
    p.add_argument('--featdf', metavar='FILE',
                   type=argparse.FileType('r'), required=True,
                   help='path to pickled features DataFrame')
    p.add_argument('--data', metavar='FILE',
                   type=argparse.FileType('w'), required=True,
                   help='output file in which to store data '
                   '(pickled DataFrame)')
    p.add_argument('--labels', metavar='FILE',
                   type=argparse.FileType('w'), required=True,
                   help='output file in which to store labels '
                   '(pickled DataFrame)')
    p.add_argument('--maps', metavar='FILE',
                   type=argparse.FileType('w'), required=True,
                   help='output file in which to store packing maps (JSON)')


# The _task_handler dictionary maps each 'command' to a (task_handler,
# parser_setup_handler) tuple.  Subparsers are initialized in __main__  (with
# the handler function's doc string as help text) and then the appropriate
# setup handler is called to add the details.
_task_handler = {'mkfeat': (handle_mkfeat, setup_parser_mkfeat),
                 'mkdata': (handle_mkdata, setup_parser_mkdata)}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='description')

    # add subparser for each task
    subparsers = parser.add_subparsers()
    for cmd, (func, p_setup) in _task_handler.items():
        p = subparsers.add_parser(cmd, help=func.__doc__)
        p.set_defaults(func=func)
        p_setup(p)

    # parse the arguments and run the handler associated with each task
    args = parser.parse_args()
    args.func(args)
