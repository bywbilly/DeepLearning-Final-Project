from __future__ import division
import os
import sys
import numpy as np
import collections
from scipy.interpolate import CubicSpline
from python_speech_features import mfcc
#from python_speech_features import delta
from python_speech_features import logfbank

Datum = collections.namedtuple('Datum', [
    'engy', 'f0',
    'slope_num', 'slope_engy', 'slope_f0',
    'length', 'pinyin', 'tone'])


def read_all():
    def _read_dataset(dataset, dirname, names):
        for name in names:
            if not name.endswith('.engy'):
                continue
            basename = name[:-len('.engy')]
            pinyin_with_tone = basename.split('_')[-1]
            pinyin = pinyin_with_tone[:-1]
            tone = int(pinyin_with_tone[-1])
            f_engy = open(os.path.join(dirname, name))
            f_f0 = open(os.path.join(dirname, basename + '.f0'))
            engy = np.array(map(float, f_engy.readlines()))
            f0 = np.array(map(float, f_f0.readlines()))
            f_engy.close()
            f_f0.close()
            assert len(engy) == len(f0)
            dataset.append(Datum(engy=engy, f0=f0,
                                 slope_num=None, slope_engy=None, slope_f0=None,
                                 length=len(engy), pinyin=pinyin, tone=tone))

    def _read_one(dirname):
        dataset = []
        os.path.walk(dirname, _read_dataset, dataset)
        return dataset

    base_dir = '../toneclassifier'
    return {
        'test': _read_one(os.path.join(base_dir, 'test')),
        'test_new': _read_one(os.path.join(base_dir, 'test_new')),
        'train': _read_one(os.path.join(base_dir, 'train')),
    }


def shuffle(datasets):
    for dataset in datasets.itervalues():
        np.random.shuffle(dataset)


def strip_zeros(datasets, epsilon=0.01):
    def _get_offset(data):
        eps = np.max(np.abs(data)) * epsilon
        for st in xrange(len(data)):
            if abs(data[st]) >= eps:
                break
        for ed in xrange(len(data)-1, -1, -1):
            if abs(data[ed]) >= eps:
                break
        return st, ed

    for dataset in datasets.itervalues():
        for i, datum in enumerate(dataset):
            st_engy, ed_engy = _get_offset(datum.engy)
            st_f0, ed_f0 = _get_offset(datum.f0)
            st = min(st_engy, st_f0)
            ed = max(ed_engy, ed_f0)
            dataset[i] = datum._replace(engy=datum.engy[st:ed+1],
                                        f0=datum.f0[st:ed+1])


def strip_zeros_by_energy(datasets, epsilon=0.01):
    for dataset in datasets.itervalues():
        for i, datum in enumerate(dataset):
            absed = np.abs(datum.engy)
            eps = np.max(absed) * epsilon
            idx = absed > eps
            dataset[i] = datum._replace(engy=datum.engy[idx],
                                        f0=datum.f0[idx],
                                        length=np.sum(idx))


def fix_length(datasets, length, f):
    def _fix_length(data):
        d = []
        chunksize = len(data) // length
        for i in xrange(length):
            st = i*chunksize
            ed = len(data) if i == length-1 else (i+1)*chunksize
            d.append(f(data[st:ed]))
        return d

    for dataset in datasets.itervalues():
        for i, datum in enumerate(dataset):
            dataset[i] = datum._replace(engy=_fix_length(datum.engy),
                                        f0=_fix_length(datum.f0))


def fix_length_by_interpolatation(datasets, length):
    for dataset in datasets.itervalues():
        for i, datum in enumerate(dataset):
            xs = np.linspace(0, 1, datum.length)
            ip_engy = CubicSpline(xs, datum.engy)
            ip_f0 = CubicSpline(xs, datum.f0)
            xs = np.linspace(0, 1, length)
            dataset[i] = datum._replace(engy=ip_engy(xs),
                                        f0=ip_f0(xs))


def calc_segmented_slope(datasets, n):
    def _slope(xs):
        k, b = np.polyfit(np.arange(len(xs)), xs, 1)
        return k

    for dataset in datasets.itervalues():
        for i, datum in enumerate(dataset):
            slope_engy = map(_slope, np.array_split(datum.engy, n))
            slope_f0 = map(_slope, np.array_split(datum.f0, n))
            dataset[i] = datum._replace(slope_engy=slope_engy,
                                        slope_f0=slope_f0,
                                        slope_num=n)

