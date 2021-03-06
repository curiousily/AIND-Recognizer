import warnings
import operator

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict,
            all_word_Xlengths: dict,
            this_word: str,
            n_constant=3,
            min_n_components=2, max_n_components=10,
            random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def _bic_score(self, log_l, p, N):
        return -2 * log_l + p * np.log(N)

    def _num_params(self, n_states, n_features):
        occupation_probs = n_states
        transition_probs = n_states * (n_states - 1)
        emission_probs = n_states * n_features * 2
        return occupation_probs + transition_probs + emission_probs

    def _get_best_n_components(self, scores):
        if len(scores) == 0:
            return self.n_constant
        else:
            return min(scores.items(), key=operator.itemgetter(1))[0]

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        N, n_features = self.X.shape
        scores = {}

        for n_components in range(self.min_n_components,
                                  self.max_n_components + 1):
            model = self.base_model(n_components)
            try:
                s = model.score(self.X, self.lengths)
                n_params = self._num_params(n_components, n_features)
                scores[n_components] = self._bic_score(s, n_params, N)
            except:
                pass
        n_comps = self._get_best_n_components(scores)
        return self.base_model(n_comps)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def _dic_score(self, log_p_i, log_p_sum, M):
        log_p_all_but_i = log_p_sum - log_p_i
        return log_p_i - log_p_all_but_i / (M - 1)

    def _get_best_n_components(self, scores):
        if len(scores) == 0:
            return self.n_constant
        else:
            score_sum = sum(scores.values())
            M = len(scores)
            dic_scores = [self._dic_score(s, score_sum, M) for s in scores.values()]
            comp_idx = np.argmax(dic_scores)
            return list(scores.keys())[comp_idx]

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on DIC scores

        scores = {}

        for n_components in range(self.min_n_components,
                                  self.max_n_components + 1):
            model = self.base_model(n_components)
            try:
                scores[n_components] = model.score(self.X, self.lengths)
            except:
                pass
        best_n_components = self._get_best_n_components(scores)
        return self.base_model(best_n_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def fit_model(self, n_components, X, y):
        model = GaussianHMM(n_components=n_components,
                covariance_type="diag",
                n_iter=1000, 
                random_state=self.random_state,
                verbose=False)
        return model.fit(X, y)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if len(self.sequences) < 3:
            return self.base_model(self.n_constant)

        kfold = KFold(n_splits=2, random_state=self.random_state)

        scores = {}

        for n_components in range(self.min_n_components,
                                  self.max_n_components + 1):

            cmp_scores = []
            for train_idx, test_idx in kfold.split(self.sequences):

                X_train, y_train = combine_sequences(train_idx, self.sequences)
                X_test, y_test = combine_sequences(test_idx, self.sequences)
                try:
                    model = self.fit_model(n_components, X_train, y_train)
                    cmp_scores.append(model.score(X_test, y_test))
                except:
                    # Probably not a good model, skip it
                    pass
            if len(cmp_scores) > 0:
                scores[n_components] = np.mean(cmp_scores)
        if len(scores) == 0:
            best_n_components = self.n_constant
        else:
            best_n_components = max(scores.items(),
                                    key=operator.itemgetter(1))[0]
        return self.base_model(best_n_components)
