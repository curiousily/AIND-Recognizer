import warnings
import operator
import math
from asl_data import SinglesData


def _score_data(model, X, y):
    try:
        return model.score(X, y)
    except:
        return -math.inf


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for X, y in test_set.get_all_Xlengths().values():
        seq_probs = {w: _score_data(m, X, y) for w, m in models.items()}
        probabilities.append(seq_probs)
        guesses.append(max(seq_probs.items(), key=operator.itemgetter(1))[0])

    return probabilities, guesses
