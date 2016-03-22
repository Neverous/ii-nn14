import pickle
import collections

WORD_LIMIT = 1024

with open('datasets/word.set', 'rb') as word_based:
    word_based_words, word_based_sentences = pickle.load(word_based)
    word_based_sentences = list([list(sentence) for sentence in word_based_sentences])
    count_words = collections.Counter()
    word_based_words_left = {}
    for sentence in word_based_sentences:
        for word in sentence:
            count_words[word] += 1

    popular_words = set([w for w, _ in count_words.most_common(WORD_LIMIT)])
    def filter_sentence(sentence, popular_words):
        for word in sentence:
            if word not in popular_words:
                return False

        return True

    word_based_sentences = list(filter(lambda sentence: filter_sentence(sentence, popular_words), word_based_sentences))
    word_based_words_left = {}
    word_based_words_left_rev = {}
    current_count = 0
    for s, sentence in enumerate(word_based_sentences):
        for w, word in enumerate(sentence):
            if word_based_words[word] not in word_based_words_left_rev:
                word_based_words_left[current_count] = word_based_words[word]
                word_based_words_left_rev[word_based_words[word]] = current_count
                current_count += 1

            word_based_sentences[s][w] = word_based_words_left_rev[word_based_words[word]]

    word_based_words = word_based_words_left
    word_based_words_rev = word_based_words_left_rev
    word_based_sentences_count = len(word_based_sentences)
    word_based_words_count = len(word_based_words)
    word_based_max_sentence_length = max([len(sentence) for sentence in word_based_sentences])
    word_based_avg_sentence_length = sum([len(sentence) for sentence in word_based_sentences]) / word_based_sentences_count
    word_based_min_sentence_length = min([len(sentence) for sentence in word_based_sentences])
    word_based_diff_sentence_length = len(set([len(sentence) for sentence in word_based_sentences]))
    print 'word-based: sentences: %d, words: %d, max-length: %d, min-length: %d, average-length: %d, different-length: %d' % (
                 word_based_sentences_count,
                 word_based_words_count,
                 word_based_max_sentence_length,
                 word_based_min_sentence_length,
                 word_based_avg_sentence_length,
                 word_based_diff_sentence_length)

if __name__ == '__main__':
    import numpy
    import matplotlib.pyplot as plt

    plt.hist([len(sentence) for sentence in word_based_sentences], 27)
    plt.xlabel('Dlugosc')
    plt.ylabel('Liczba zdan')
    plt.savefig('pictures/word_length_distribution.png')
    plt.clf()
    plt.cla()
    plt.close()

    _, bins, _ = plt.hist([ord(letter) - ord('a') for sentence in word_based_sentences for word in sentence for letter in word_based_words[word] if letter >= 'a'], 26)
    bin_centers = 0.5 * numpy.diff(bins) + bins[:-1]
    for letter, x in zip([chr(i + ord('a')) for i in xrange(26)], bin_centers):
        plt.annotate(letter, xy=(x, 0), xycoords=('data', 'axes fraction'),
                     xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    plt.ylabel('Liczba wystapien')
    plt.xlim(0, 26)
    plt.savefig('pictures/word_letters_distribution.png')
    plt.clf()
    plt.cla()
    plt.close()
