import codecs
import logging
import os
import os.path as P

from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch and show training parameters '''

    def __init__(self, savedir):
        self.savedir = savedir
        self.epoch = 0
        os.makedirs(self.savedir, exist_ok=True)

    def on_epoch_end(self, model):
        savepath = os.path.join(self.savedir, "model_fastText_web_kw_sm{}_epoch.gz".format(self.epoch))
        model.save(savepath)
        print(
            "Epoch saved: {}".format(self.epoch + 1),
            "Start next epoch ... ", sep="\n"
            )
        if os.path.isfile(os.path.join(self.savedir, "model_fastText_web_kw_sm{}_epoch.gz".format(self.epoch - 1))):
            print("Previous model deleted ")
            os.remove(os.path.join(self.savedir, "model_fastText_web_kw_sm{}_epoch.gz".format(self.epoch - 1)))
        self.epoch += 1


class SentenceIter:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        curr_dir = P.dirname(P.abspath(__file__))
        with codecs.open(P.join(curr_dir, self.filename), "r", 'utf-8', errors='replace') as fin:
            for line in fin:
                words_bad = line[:-1].split(" ")  # Yielding this causes a segfault
                words_good = line.rstrip().split(' ')  # Yielding this is OK
                logging.debug('words_bad: %r', words_bad)
                logging.debug('words_good: %r', words_good)
                logging.debug('---')
                yield words_good


def main():
   logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

   num_workers = os.cpu_count()
   model = FastText(
        # SentenceIter("data.txt"),  # This always segfaults
        SentenceIter("data-example.txt"),  # This works when the iterator yields words_good
        sg=1,
        size=100,
        window=3,
        min_count=5,
        workers=num_workers,
        iter=5,
        negative=20,
        # callbacks=[EpochSaver("./checkpoints/fasttext_eng_tweets")]
    )


if __name__ == "__main__":
    main()
