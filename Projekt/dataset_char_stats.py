import pickle
import collections

with open('datasets/char.set', 'rb') as char_based:
    char_based_sentences = list(pickle.load(char_based))
    char_based_sentences_count = len(char_based_sentences)
    char_based_letters_rev = dict(enumerate(sorted(set([letter for sentence in char_based_sentences for letter in sentence]))))
    char_based_letters = dict([(b, a) for a, b in char_based_letters_rev.items()])
    char_based_letters_count = len(char_based_letters)
    char_based_max_sentence_length = max([len(sentence) for sentence in char_based_sentences])
    char_based_avg_sentence_length = sum([len(sentence) for sentence in char_based_sentences]) / char_based_sentences_count
    char_based_min_sentence_length = min([len(sentence) for sentence in char_based_sentences])
    char_based_diff_sentence_length = len(set([len(sentence) for sentence in char_based_sentences]))
    print 'Char-based: sentences: %d, letters: %d, max-length: %d, min-length: %d, average-length: %d, different-length: %d' % (
                 char_based_sentences_count,
                 char_based_letters_count,
                 char_based_max_sentence_length,
                 char_based_min_sentence_length,
                 char_based_avg_sentence_length,
                 char_based_diff_sentence_length)

    buckets = collections.defaultdict(lambda: list())
    for sentence in char_based_sentences:
        buckets[len(sentence)].append(sentence)

    char_based_length_choice = 0
    char_based_length_choice_count = 0
    current_count = 0
    for length, bucket in reversed(sorted(buckets.items())):
        current_count += len(bucket)
        if length * current_count > char_based_length_choice * char_based_length_choice_count:
            char_based_length_choice = length
            char_based_length_choice_count = current_count

    print 'Most dense length: %d x %d => %d' % (char_based_length_choice, char_based_length_choice_count, char_based_length_choice * char_based_length_choice_count)

if __name__ == '__main__':
    import numpy
    import matplotlib.pyplot as plt

    plt.hist([len(sentence) for sentence in char_based_sentences], 424)
    plt.xlabel('Dlugosc')
    plt.ylabel('Liczba zdan')
    plt.savefig('pictures/char_length_distribution.png')
    plt.clf()
    plt.cla()
    plt.close()

    _, bins, _ = plt.hist([char_based_letters[letter] for sentence in char_based_sentences for letter in sentence], 63)
    bin_centers = 0.5 * numpy.diff(bins) + bins[:-1]
    for letter, x in zip([char_based_letters_rev[i] for i in xrange(63)], bin_centers):
        plt.annotate(letter, xy=(x, 0), xycoords=('data', 'axes fraction'),
                     xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    plt.ylabel('Liczba wystapien')
    plt.xlim(0, 63)
    plt.savefig('pictures/char_letters_distribution.png')
    plt.clf()
    plt.cla()
    plt.close()
