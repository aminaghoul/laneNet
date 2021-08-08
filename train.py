# from model import ConvLSTM
from contextlib import contextmanager
from datetime import timedelta
from inspect import stack
from itertools import chain
from signal import signal, SIGALRM, alarm
from time import time

from torch.utils.data import Dataset

from main import NS


@contextmanager
def do_something(what):
    print(what + '...', end='', flush=True)
    begin = time()
    yield
    print('\r' + what + '...done (in %s)' % timedelta(seconds=time() - begin))


class SuperNS(Dataset):
    # TODO: Add more parameters as well as more
    #  attributes to this.
    def __init__(self):
        self.__ns = list(NS.every_map('config.yml'))
        self.__items = list(chain.from_iterable(self.__ns))
        self.helper = self.__ns[0].helper

    def __len__(self):
        # This is the list of every element of
        #  every datasets
        return len(self.__items)

    def __getitem__(self, item):
        # This is the list of every element of
        #  every datasets
        return self.__items[item]


def show_trace(interval, limit=None):
    """
    Warning: uses alarms

    :param interval: How much seconds
    :param limit: Maximum amount of rows
    :return:
    """

    def handler(*args):
        print('Current state of the stack:')
        for args in enumerate(stack()[1:][:limit]):
            print(*args, sep=' -> ')
        print('...........................')
        alarm(interval)

    signal(SIGALRM, handler)
    alarm(interval)
