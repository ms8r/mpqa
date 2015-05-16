#!/usr/bin/env python

"""
Processes MPQA corpus
"""

from __future__ import division, print_function, unicode_literals
import os
from collections import namedtuple
from itertools import islice
import pandas as pd
import spacy.en
import argparse
import logging


FEAT_COLS = [
    'word',         # the subj. clue word (id)
    'word_',        # the subj. clue word (string)
    'pos',          # part-of-speach (id)
    'pos_',         # part-of-speach (string)
    'before',       # token preceding word (id)
    'before_',      # token preceding word (string)
    'after',        # token following word (id)
    'after_',       # token following word (string)
    'context',      # tuple (before, word, after)
    'context_',     # tuple (before, word, after) with strings
    'pre_neg',      # negation in front of token
    'post_neg',     # negation after token
    'pri_pol',      # prior polarity
    'rel',          # reliability class {strongsubj, weaksubj}
    'c_pol',        # contextual polarity
    'is_int',       # is intensifier (boolean)
    'prec_int',     # preceded by intensifier (boolean)
    'prec_adj',     # preceded by adjective (boolean)
    'prec_adv',     # preceded by adverb (boolean)
    'topic',        # topic (if available)
]


Annot = namedtuple('Annot', ['idnum', 'start', 'end', 'ref',
                             'kind', 'gate', 'attr'])
GateSent = namedtuple('GateSent', ['start', 'end'])
SubClue = namedtuple('SubClue', ['rel', 'pri_pol', 'stemmed'])
Polar = namedtuple('Polar', ['token', 'rel', 'pri_pol', 'c_pol'])

logging.basicConfig(level=logging.INFO)


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

    _negations = set(['no', 'not', 'neither', 'nor', 'nobody', 'none', 'n\'t'])
    _doub_negs = set(['not only'])
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

    @property
    def features(self):
        """
        Contructs feature vectors as a list of dicts, keyed by the items in
        the global `FEAT_COLS`.
        """
        # TODO: map each item of FEAT_COLS to a helper function that constructs
        # the corresponding feature

        if self._features is not None:
            return self._features

        voc = self._nlp.vocab

        feats_list = []
        num_toks = len(self.tokens)
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
            pre = self.tokens[ti - 1] if ti > 0 else None
            feats['before'] = voc[pre.lower_].id if pre else -1
            feats['before_'] = pre.lower_ if pre else ''
            post = self.tokens[ti + 1] if ti < num_toks else None
            feats['after'] = voc[post.norm_].id if post else -1
            feats['after_'] = post.norm_ if post else ''
            feats['context'] = (feats['before'], feats['word'], feats['after'])
            feats['context_'] = (feats['before_'], feats['word_'],
                                 feats['after_'])
            feats['prec_int'] = 1 if pre and self._in.lookup(pre) else 0
            feats['prec_adj'] = 1 if pre and pre.pos_ == 'ADJ' else 0
            feats['prec_adv'] = 1 if pre and pre.pos_ == 'ADV' else 0
            feats['is_int'] = 1 if self._in.lookup(p.token) else 0
            # check for negation:
            if ti == 0:
                feats['pre_neg'] = 0
            else:
                pre_start = max(0, ti - self._neg_span)
                pre = [s.lower_ for s in islice(self.tokens, pre_start, ti)]
                bigrams = [' '.join(s) for s in zip(pre, pre[1:])]
                pre = set(pre + bigrams)
                feats['pre_neg'] = (
                        1 if pre.intersection(self._negations) and not
                        pre.intersection(self._doub_negs) else 0)
            if ti == num_toks - 1:
                feats['post_neg'] = 0
            else:
                post_end = min(num_toks, ti + self._neg_span)
                post = [s.lower_ for s in islice(self.tokens, ti, post_end)]
                bigrams = [' '.join(s) for s in zip(post, post[1:])]
                post = set(post + bigrams)
                feats['post_neg'] = (
                        1 if post.intersection(self._negations) and not
                        post.intersection(self._doub_negs) else 0)

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


def make_pack_map(df, columns):
    """
    Creates a mapping of in int ids in `columns` (list) of DataFrame df onto
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
    Takes integer ids in `columns` (list) fo df (pandas DataFrame) and maps
    them on a set of consequtive ints. Returns a DataFrame with `columns`.
    """
    pack_map = make_pack_map(df, columns)
    df_packed = pd.DataFrame(columns=columns)
    for c in columns:
        df_packed[c] = df[c].map(lambda k: pack_map[k])

    return df_packed


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
    Creates a feature DataFrame and stores same as JSON file.
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

    # append columns with packed vocabulary references:
    pack_cols = pack_df(df, ['word', 'before', 'after'])
    df['pword'] = pack_cols['word']
    df['pbefore'] = pack_cols['before']
    df['after'] = pack_cols['after']

    # save df:
    if args.format == 'json':
        # drop text token columns as they screw up json encoding:
        keep_cols = [c for c in FEAT_COLS if not c.endswith('_')]
        keep_cols += ['pword', 'pbefore', 'pafter']
        df[keep_cols].to_json(outfile)
    elif args.format == 'pickle':
        df.to_pickle(outfile)
    else:
        logging.error('unknown output format %s', args.format)


def setup_parser_mkfeat(p):
    p.add_argument('--mpqa', default='.', help='path to MPQA root directory; '
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
    p.add_argument('--format', choices=['pickle', 'json'], default='json',
                    help='format in which to output DataFrame; defaults to '
                    'json; only word token ids (rather than the strings) '
                    'will be retained; the ids can be mapped back to strings '
                    'by using spaCy\'s English vocabulary (see '
                    'http://honnibal.github.io/spaCy/)')
    p.add_argument('--output', metavar='FILE',
                   type=argparse.FileType('w'), required=True,
                   help='output file in which to store DataFrame')


# The _task_handler dictionary maps each 'command' to a (task_handler,
# parser_setup_handler) tuple.  Subparsers are initialized in __main__  (with
# the handler function's doc string as help text) and then the appropriate
# setup handler is called to add the details.
_task_handler = {'mkfeat': (handle_mkfeat, setup_parser_mkfeat), }


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
