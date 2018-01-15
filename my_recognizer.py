import warnings
from asl_data import SinglesData


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
    #
    for x, lengths in test_set.get_all_Xlengths().values():
        dict_logL = {}
        # check each model and get the score
        for word, model in models.items():
            try:
                dict_logL[word] = model.score(x, lengths)
            except:
                dict_logL[word] = float('-inf')

        # append the dictionary into the list
        probabilities.append(dict_logL)
        # choose the max score into guesses
        guesses.append(max(dict_logL, key=dict_logL.get))

    return probabilities, guesses
