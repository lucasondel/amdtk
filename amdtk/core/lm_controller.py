
"""Set of operations for training and using the  Bayesian LM."""

import numpy as np
import string
import pywrapfst as fst
from ..models.hpyp import EMPTY_CONTEXT
from ..models.hpyp import VOCAB_START
from ..models.hierarchical_language_model import SPECIAL_VOCAB
from ..models.hpyp import SEQ_END


class LMParamsStringBadlyFormatted(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def wordTokenize(text):
    return text.lower().strip().split()


def removeSentenceMarker(text):
    """remove leading or trailing sentence_markers
    (or anywhere else)"""
    sentence_marker = '</s>'
    return text.replace(sentence_marker, '')


def parseLMParams(str):
    """Parse the HPYP parameters formatted as a string.

    The string format of the HPYP is:
      d0,c0:c1,d1:c2,d2:...
    where d0 is the discount and c0 the concentration of the first level
    of the hierarchy.

    Parameters
    ----------
    str : string
        HPYP paramaters as a string.

    Returns
    -------
    params : list
        List of tuple (discount, concentration). The first element of
        the list corresponds to the first level of the hierarchy.

    """

    def parse(s):
        tokens = s.split(',')
        print(tokens)
        if len(tokens) != 2:
            raise LMParamsStringBadlyFormatted('Expect 2 parameters: discount '
                                               'and concentration.')
        return float(tokens[0]), float(tokens[1])
    return [parse(s) for s in str.split(':')]


def textToInt(vocab, text, tokenize=wordTokenize):
    """Convert space separated tokens to integer.

    Parameters
    ----------
    vocab : dict
        Vocabulary to use while processing the text. If a token is not
        in the vocabulary it will be removed silently.
    text: string
        Text to convert.
    tokenize: function
        The function used to perform tokenization of the input.

    Returns
    -------
    text_int: list
        The text as a list of integer.

    """
    data_int = []
    for token in tokenize(text):
        try:
            data_int.append(vocab[token])
        except ValueError:
            pass
    data_int.append(SEQ_END)
    return data_int


def getVocabFromText(data_text, remove_punctuation=True,
                     tokenize=wordTokenize):
    """Get vocabulary from the given text.

    Parameters
    ----------
    data_text: list of strings
        Text data from which to extract the vocabulary.
    remove_punctuation: boolean
        If true, do not add the punctuation in the vocabulary
        (default: True).
    tokenize: function
        The function used to perform tokenization of the input.

    Returns
    -------
    vocab: list
        List of unique word/element in the text.
    """

    word_set = set([])
    for line in data_text:
        line = removeSentenceMarker(line)
        if remove_punctuation:
            translator = line.maketrans({key: None for key in
                                        string.punctuation})
            line = line.translate(translator)

        for token in tokenize(line):
            word_set.add(token)
    vocab = {word: id
             for id, word in enumerate(sorted(list(word_set)), VOCAB_START)}
    return vocab


def prepareText(data_text, vocab=None, remove_punctuation=True,
                tokenize=wordTokenize):
    """Prepare text to build HPYP LM.

    Parameters
    ----------
    data_text: list of strings
        Text data to process.
    vocab: list
        List of unique word in the text. If not provided, the vocabulary
        will be extracted from the given text.
    remove_punctuation: boolean
        If true, do not process the punctuation (default: True).
    tokenize: function
        The function used to perform tokenization of the input.

    Returns
    -------
    data_int: list of int
        Data as expected by other LM tool.

    """

    data_int = []

    if vocab is None:
        vocab = getVocabFromText(data_text, tokenize)

    for line in data_text:
        line = removeSentenceMarker(line)
        if remove_punctuation:
            translator = line.maketrans({key: None for key in
                                        string.punctuation})
            line = line.translate(translator)
            data_int.append(textToInt(vocab, line, tokenize))

    return data_int


def getContext(seq, n):
    """Transform the sequence into a "context" format as expected by
    HPYP model.

    Parameters
    ----------
    seq: list
        Sequence from where to read the context.
    n: int
        Length of the context.

    Returns
    -------
    context: list
        n-gram context.

    """
    if n == 0:
        context = EMPTY_CONTEXT
    else:
        length = len(seq)
        context_ix = max(0, length-n)
        context = seq[context_ix:]
        if length - n < 0:
            pad_length = n - (length-context_ix)
            padding = [-1] * pad_length
            context = padding + context
    return tuple(context)


def initNgramLM(model, data_int):
    """Initialize n-gram LM for HPYP.

    Parameters
    ----------
    model: :class:`HierarchicalPitmanYorProcess`
        Hierarchy of PYP to initialize.
    data_int: list of int
        Preprocessed text.

    """
    level = model.order
    for line_int in data_int:
        for i, token_dish in enumerate(line_int):

            # Extract the context from the history of the current token.
            context = getContext(line_int[:i], level)

            # Try to get the restaurant or create it in case of failure.
            try:
                restaurant = model[level][context]
            except KeyError:
                # we want to create a new restaurant
                restaurant = model.addRestaurant(level, context)

            # Serve the dish in the previously obtained restaurant.
            restaurant.serveDish(token_dish, remove_customer=False)


def sampleNgramLM(model, data_int, keep_add=False):
    """Sample a seating arrangement using Gibbs sampling.

    Parameters
    ----------
    model: :class:`HierarchicalPitmanYorProcess`
        Hierarchy of PYP to initialize.
    data_int: list of int
        Preprocessed text
    keep_add: boolean
        If true, do not remove customer but keep adding them to the
        restaurant.

    """
    for line_int in data_int:
        for i, token_dish in enumerate(line_int):

            # Extract the context from the history of the current token.
            context = getContext(line_int[:i], model.order)

            # restaurant = model[model.order][context]
            try:
                restaurant = model[model.order][context]
            except KeyError:
                restaurant = model.addRestaurant(model.order, context)
            restaurant.serveDish(token_dish, remove_customer=not keep_add)


def resampleNgramLM(model, data1_int, data2_int):
    """Resample a seating arrangement using Gibbs sampling.

    Parameters
    ----------
    model: :class:`HierarchicalPitmanYorProcess`
        Hierarchy of PYP to initialize.
    data1_int: list of int
        Preprocessed text corresponding to the current seating arrangement.
    data2_int: list of int
        Preprocessed text to use for resampling.

    """
    for i in range(len(data1_int)):
        line1_int = data1_int[i]
        line2_int = data2_int[i]

        # We remove all the dishes from the previous utterance.
        for j, token_dish in enumerate(line1_int):
            # Extract the context from the history of the current token.
            context = getContext(line1_int[:j], model.order)

            # Get the restaurant for the given context.
            try:
                restaurant = model[model.order][context]
            except KeyError:
                restaurant = model.addRestaurant(model.order, context)

            # Remove a customer eating this dish.
            restaurant.removeCustomer(token_dish)

        # Now add the customers from the new text
        for j, token_dish in enumerate(line2_int):
            # Extract the context from the history of the current token.
            context = getContext(line2_int[:j], model.order)

            # Get the restaurant for the given context.
            try:
                restaurant = model[model.order][context]
            except KeyError:
                restaurant = model.addRestaurant(model.order, context)
            restaurant.serveDish(token_dish, remove_customer=False)


def NgramLMLogLikelihood(model, data_int):
    """Compute the n-gram LM log-likelihood.

    Parameters
    ----------
    model: :class:`HierarchicalPitmanYorProcess`
        Hierarchy of PYP to initialize.
    data_int: list of int
        Preprocessed text

    Returns
    -------
    llh: float
        logarithm of the predictive probability of the text given the
        model.

    """
    llh = 0
    for line_int in data_int:
        for i, token_dish in enumerate(line_int):
            # Extract the context from the history of the current token.
            context = getContext(line_int[:i], model.order)
            llh += np.log(model.predictiveProbability(model.order, context,
                                                      token_dish))
    return llh


def __walkFst(f, state, id2label, path):
    arcs = list(f.arcs(state))

    # If there are no more outgoing arcs finish the recursion.
    if len(arcs) == 0:
        return

    probs = np.array([np.exp(-float(arc.weight.string)) for arc in arcs])

    # Because of numerical precision issues, it is sometimes necessary to
    # renormalize the probabilites.
    probs /= probs.sum()

    arc = np.random.choice(arcs, p=probs)

    # Add the label to the path except if it is an epsilon label.
    if arc.ilabel != 0:
        path.append(id2label[arc.ilabel])

    # Follow the chosen arc.
    __walkFst(f, arc.nextstate, id2label, path)


def samplePathFromFst(fst_lattice, id2label):
    """Sample path from a lattice.

    Parameters
    ----------
    fst_lattice : fst.Fst
        Lattice in OpenFst format.
    id2labels : mapping arc label id to human readable labels.

    Returns
    -------
    path : list
        Sequence of (human readable) labels.

    """
    # Transform fst_lattice into a stochastic FST.
    stoc_fst_lattice = fst.push(fst_lattice, push_weights=True,
                                remove_total_weight=True)

    # Random walk on the stochastic FST.
    path = []
    __walkFst(stoc_fst_lattice, stoc_fst_lattice.start, id2label, path)

    return path




