from math import log

EMPTY_WORD = -1
EMPTY_CONTEXT = tuple([EMPTY_WORD])
EMPTY_WORD_LABEL = "<0>"
EPS_LABEL, EPS = "<eps>", 0
PHI_LABEL, PHI = "<phi>", 1
SEQ_END_LABEL, SEQ_END = "</s>", 2
SPECIAL_VOCAB = {EPS_LABEL: EPS,
                 PHI_LABEL: PHI,
                 SEQ_END_LABEL: SEQ_END}
VOCAB_START = len(SPECIAL_VOCAB)


class HierarchicalLanguageModel(object):
    """
        Base class for any kind of hierarchical language models.
    """

    def __init__(self, order, vocab=None):
        """

        :param order: Order of the hierarchical language model
        """
        self.hierarchy = [{} for i in range(order + 1)]
        self.vocab = vocab
        self._order = order

    @property
    def order(self):
        return self._order

    def __getitem__(self, key):
        # Safety checks.
        if type(key) != int:
            raise TypeError()
        if key < 0 or key >= len(self.params):
            raise IndexError()
        return self.hierarchy[key]

    @property
    def vocabSize(self):
        if self.vocab is None:
            raise RuntimeError("No given vocab!")
        return len(self.vocab)

    def getFullVocab(self):
        full_vocab = list(SPECIAL_VOCAB.items())
        full_vocab.extend(list(self.vocab.items()))

        return sorted(full_vocab, key=lambda x: x[1])

    def predictiveProbability(self, level, context, word):
        """ Probability of a word in a given context

        :param level: language model order to use
        :param context: given context as word id tuple
        :param word: word id for requested word
        :return: probability of word
        """
        pass

    def fallbackProbability(self, level, context):
        """ Probability to do a backoff to a shortened context

        :param level: language model order to use
        :param context: given context as word id tuple
        :return: probability of backing off
        """
        pass

    def _getContexts(self, level):
        def context_key(context):
            return (context[0] + 1) + (
                100*context_key(context[1:]) if len(context) > 1 else 0)

        return sorted(self.hierarchy[level].keys(), key=context_key)

    def _getWordsInContext(self, level, context):
        return self.hierarchy[level][context].getWords()

    def _formatOpenFstLine(self, begin_id, end_id,
                           in_label, out_label, weight):
        """
        :param level: desired language model order
        :param context_to_id: Dictionary containing a number for each context.
            Used to get the id for the followup context.
        :param context: n-gram context id tuple defining the BEGIN_ID state
        :param word_id: The word id transduced by the arc
        :param word_label: same as above, but as string LABEL
        :return: string with output format:
                    "BEGIN_ID\tEND_ID\tLABEL\tLABEL\tWEIGHT"
        """
        return "{b}\t{e}\t{il}\t{ol}\t{w}\n".format(
            b=begin_id,
            e=end_id,
            il=in_label,
            ol=out_label,
            w=weight)

    def _fallbackTransition(self, context_to_id, start_context):
        end_context = (start_context[1:]
                       if len(start_context) > 1 else EMPTY_CONTEXT)
        return self._formatOpenFstLine(
            context_to_id[start_context],
            context_to_id[end_context],
            PHI_LABEL,
            EPS_LABEL,
            -log(self.fallbackProbability(len(start_context), start_context))
        )

    def exportGrammarFSA(self, level=None):
        """
        Converts the language model into a Grammar FSA in OpenFst text format

        :return: grammar_fsa: grammar fsa in openFst format, i.e.:
                            list of strings each representing a line of the
                            text file
                            format: "BEGIN_ID\tEND_ID\tLABEL\tLABEL\tWEIGHT"
        """
        if level is None:
            level = self.order

        grammar_fsa = []
        # set up the dict containing all possible contexts
        context_to_id = {context: id for id, context in enumerate(
            [context for current_level in range(0, level + 1)
             for context in self._getContexts(current_level)], 1)}

        word_id_to_label = {
            word_id: word_label for (word_label, word_id) in self.vocab.items()
        }
        word_id_to_label.update({SEQ_END: SEQ_END_LABEL})
        # add start state
        seq_start_context = tuple([EMPTY_WORD]*(self.order))
        grammar_fsa.append("0\t{e}\t{eps}\t{eps}\n".format(
            e=context_to_id[seq_start_context], eps=EPS_LABEL))

        # add transitions for each state in LM
        for start_context in context_to_id.keys():
            if start_context != EMPTY_CONTEXT:
                start_id = context_to_id[start_context]
                # add transitions for all words in current context
                for word_id in self._getWordsInContext(len(start_context),
                                                       start_context):
                    if word_id == SEQ_END:
                        end_context = seq_start_context
                    else:
                        shorten_context = (len(start_context) == self.order or
                                           start_context == EMPTY_CONTEXT)
                        end_context = (start_context[int(shorten_context):]
                                       + (word_id,))
                    try:
                        end_id = context_to_id[end_context]
                        label = word_id_to_label[word_id]
                        weight = -log(self.predictiveProbability(
                            len(start_context), start_context, word_id))
                        grammar_fsa.append(self._formatOpenFstLine(
                            start_id, end_id, label, label, weight)
                        )
                    except KeyError:
                        pass

                # add fallback (phi) transitions
                try:
                    grammar_fsa.append(self._fallbackTransition(
                        context_to_id, start_context))
                except KeyError:
                    pass
            else: # the empty context!
                for word_id in word_id_to_label.keys():
                    end_context = ((word_id,) if word_id != SEQ_END
                                   else seq_start_context)
                    start_id = context_to_id[EMPTY_CONTEXT]
                    try:
                        end_id = context_to_id[end_context]
                    except KeyError:
                        end_id = context_to_id[EMPTY_CONTEXT]

                    label = word_id_to_label[word_id]
                    weight = -log(self.predictiveProbability(0, EMPTY_CONTEXT,
                                                             word_id))
                    grammar_fsa.append(self._formatOpenFstLine(
                        start_id, end_id, label, label, weight
                    ))

        # add final state
        grammar_fsa.extend("{cid}\n".format(
            cid=context_to_id[seq_start_context]))

        return grammar_fsa
