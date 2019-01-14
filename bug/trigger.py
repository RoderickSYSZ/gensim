import os
import os
import logging

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
    def __iter__(self):
        import codecs
        import os.path as P
        curr_dir = P.dirname(P.abspath(__file__))
        with codecs.open(P.join(curr_dir, "data.txt"), "r", 'utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i > 100:
                    break
                yield line[:-1].split(" ")

if __name__ == "__main__":

   logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

   num_workers = os.cpu_count()
   model = FastText(
        SentenceIter(),
        sg=1,
        size=100,
        window=3,
        min_count=5,
        workers=num_workers,
        iter=5,
        negative=20,
        # callbacks=[EpochSaver("./checkpoints/fasttext_eng_tweets")]
    )
