import logging
from typing import List

from dataio_utils.helper import build_vocab
from gpu_env import APP_NAME, ModeKeys
import tensorflow as tf
import json
import os


class BaseDataIO:
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(APP_NAME)
        self.vocab = build_vocab(args.word_embedding_files)
        self.pad_id = self.vocab.get_id(self.vocab.pad_token)
        self.unk_id = self.vocab.get_id(self.vocab.unk_token)
        self.start_token_id = self.vocab.get_id(self.vocab.start_token)
        self.stop_token_id = self.vocab.get_id(self.vocab.stop_token)
        self.datasets = {}

    def next_batch(self, batch_size: int, mode: ModeKeys):
        raise NotImplementedError

    def load_data(self, file_paths: str, mode: ModeKeys):
        dataset = tf.data.TextLineDataset(file_paths) \
            .shuffle(5000) \
            .batch(self.args.batch_size)
            # .prefetch(self.args.batch_size * 5)

        self.logger.info('loading data for %s' % mode.name)

        return dataset

    def make_mini_batch(self, data):
        raise NotImplementedError
