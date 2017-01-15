from __future__ import division
import os
import sys
import numpy as np
import collections
from scipy.interpolate import CubicSpline
from python_speech_features import mfcc
#from python_speech_features import delta
from python_speech_features import logfbank
from scipy import signal

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

def dp_erase_noise(datasets):
    for dataset in datasets.itervalues():
        for ii, datum in enumerate(dataset):
            length = len(datum.f0)
            #print datum
            f = [[[[1000000000, [], False] for k in xrange(length + 1)] for j in xrange(length + 1)] for i in xrange(length)]
            for i in xrange(length):
                f[i][0][0][0] = 0
                f[i][0][0][2] = True
            f[0][1][1][0] = 0
            f[0][1][1][1] = [0]
            f[0][1][1][2] = True
            for i in xrange(0, length - 1):
                for j in xrange(0, i + 1):
                    for k in xrange(0, i + 2):
                        if f[i][j][k][2] == False:
                            continue
                        if f[i + 1][j][k][2]:
                            if f[i][j][k][0] < f[i + 1][j][k][0]:
                                f[i + 1][j][k][0] = f[i][j][k][0]
                                f[i + 1][j][k][1] = f[i][j][k][1]
                        else:
                            f[i + 1][j][k] = f[i][j][k]
                            f[i + 1][j][k][2] = True
                        if f[i + 1][j + 1][i + 1][2]:
                            pre = datum.f0[i]
                            if k != 0:
                                pre = datum.f0[k - 1]
                            if (f[i + 1][j + 1][i + 1][0] > f[i][j][k][0] + (datum.f0[i] - pre) ** 4 + (i - (k - 1)) ** 2):
                                f[i + 1][j + 1][i + 1][0] = f[i][j][k][0] + (datum.f0[i] - pre) ** 4 + (i - (k - 1)) ** 2 
                                f[i + 1][j + 1][i + 1][1] = f[i][j][k][1] + [i]
                        else:
                            pre = datum.f0[i]
                            if k != 0:
                                pre = datum.f0[k - 1]
                            f[i + 1][j + 1][i + 1][0] = f[i][j][k][0] + (datum.f0[i] - pre) ** 2 + (i - (k - 1)) ** 1.5
                            f[i + 1][j + 1][i + 1][1] = f[i][j][k][1] + [i]
                            f[i + 1][j + 1][i + 1][2] = True
                        #flucation = (datum.f0[i - 1] - pre) ** 4 + (i - (k - 1)) ** 2
                        #if f[i - 1][j][k] != 1000000000:
                        #    f[i][j][k] = f[i - 1][j][k]
                        #if f[i - 1][j - 1][k][0] + flucation < f[i][j][i][0]:
                        #    #print i - 1, j - 1
                        #    #print "mie"
                        #    f[i][j][i][0] = f[i - 1][j - 1][k][0] + flucation
                        #    f[i][j][i][1] = f[i - 1][j - 1][k][1] + [i - 1]
                        #    #print f[i][j][i]
            index = []
            Min = 1000000000
            pos = -1
            for j in xrange(int(length / 4), length + 1):
                for k in xrange(1, length + 1):
                    #print j, k
                    #print f[length][j][k]
                    if f[length - 1][j][k][2] and f[length - 1][j][k][0] < Min:
                        pos = j
                        #Min = f[length - 1][i][k][0]
                        Min = f[length - 1][j][k][0]
                        index = f[length - 1][i][k][1]
            #print pos, length
            #print pos - length
            #print index
            #assert(len(index) != 0)
            if len(index) <= 10:
                index = [i for i in xrange(length)]
            assert(len(index) >= 3)
            dataset[ii] = datum._replace(f0=datum.f0[index], engy=datum.engy[index], length=len(index)) 

        print "miaomiaomiao"



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


def fix_length_by_interpolatation(datasets, length, mfcc_len):
    for dataset in datasets.itervalues():
        for i, datum in enumerate(dataset):
            xs = np.linspace(0, 1, datum.length)
            ip_engy = CubicSpline(xs, datum.engy)
            ip_f0 = CubicSpline(xs, datum.f0)
            xs = np.linspace(0, 1, length)
            xs_mfcc = np.linspace(0, 1, mfcc_len)
            dataset[i] = datum._replace(engy=ip_engy(xs_mfcc),
                                        f0=ip_f0(xs))

def fix_length_by_interpolatation_engy(datasets, length):
    for dataset in datasets.itervalues():
        for i, datum in enumerate(dataset):
            xs = np.linspace(0, 1, datum.length)
            ip_engy = CubicSpline(xs, datum.engy)
            xs = np.linspace(0, 1, length)
            dataset[i] = datum._replace(engy=ip_engy(xs),
                                        )
def data_augmentation(datasets, key = 'train'):
    new = []
    for i, datum in enumerate(datasets[key]):
        new.append(datum)
        new.append(datum._replace(engy=datum.engy + np.random.normal(size = datum.engy.shape)))

    datasets[key] = new

def low_pass_filter(datasets):
    for dataset in datasets.itervalues():
        for i, datum in enumerate(dataset):
            b, a = signal.butter(5, 0.1)
            #dataset[i] = datum._replace(engy = signal.filtfilt(b, a, datum.engy))
            dataset[i] = datum._replace(engy = signal.lfilter(b, a, datum.engy))
        


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

