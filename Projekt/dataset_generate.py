# -*- encoding: utf-8 -*-

import re
import glob
import lxml.etree
import unicodedata
import string
import pickle

class XMLCorpusCleaner(object):
    filter_out = re.compile(ur"[^\w ]+", re.UNICODE)
    match_sentences = re.compile(ur"[A-Z][^\.!?]*[\.!?]", re.UNICODE)
    charmap = dict((ord(a), b) for a, b in zip(u'ąćęłńóśżźĄĆĘŁŃÓŚŻŹ', u'acelnoszzACELNOSZZ'))

    def __init__(self):
        self.sentences = set()
        pass

    def feed(self, files, tag_filter):
        start_sentences = len(self.sentences)
        for textfile in files:
            with open(textfile, 'rb') as fin:
                context = lxml.etree.iterparse(fin)
                self._fast_iter(context, tag_filter, self._import_sentences)
                print '\r', len(self.sentences) - start_sentences,

        print '\rread', len(self.sentences) - start_sentences, 'valid sentences'

    def _fast_iter(self, context, tag_filter, func, *args, **kwargs):
        for event, elem in context:
            if event == 'end' and tag_filter(elem.tag):
                func(elem, *args, **kwargs)

            elem.clear()
            try:
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
            except:
                pass

        del context

    def _import_sentences(self, elem):
        text = unicode(elem.text)
        text = text.translate(self.charmap)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore')
        text = text.replace('~', '').replace('|', '').replace('=', '').replace('*', '').replace('_', '')
        if not text:
            return

        self.sentences.update(filter(lambda x: 5 < x.count(' '), map(string.lower, self.match_sentences.findall(text))))

    def get_sentences(self):
        return sorted(self.sentences)

if __name__ == '__main__':
    corpus = XMLCorpusCleaner()
    corpus.feed(glob.glob('NKJP-PodkorpusMilionowy-1.2/*/text.xml'), lambda tag: tag.endswith('}ab'))

    valid_sentences = corpus.get_sentences()

    with open('datasets/char.set', 'wb') as char_based:
        char_based_sentences = tuple(valid_sentences[:45000])
        print "Char based sentences: ", len(char_based_sentences)
        pickle.dump(char_based_sentences, char_based)

    with open('datasets/word.set', 'wb') as word_based:
        word_based_sentences = sorted(
                set(map(lambda sentence: ' '.join(XMLCorpusCleaner.filter_out.sub(' ', sentence).split()),
                    valid_sentences))
                )[:45000]

        word_based_words = dict(enumerate(sorted(set([word for words in map(lambda sentence: sentence.split(), word_based_sentences) for word in words]))))
        word_based_words_rev = dict([(b, a) for a, b in word_based_words.items()])
        word_based_sentences = tuple(map(lambda sentence: tuple(map(lambda word: word_based_words_rev[word], sentence.split())),
            word_based_sentences))

        print "Words: ", len(word_based_words)
        print "Word based sentences: ", len(word_based_sentences)

        pickle.dump([word_based_words, word_based_sentences], word_based)
