#!/usr/bin/env python
from PIL import Image
import numpy
import requests
import StringIO
import os
import re
import sqlalchemy
import sqlalchemy.exc
import unicodedata
import numpy as np
import random
from six.moves import range

try:
    engine = sqlalchemy.create_engine('mysql://ro:rj5XYM4V9QKwy5XScdG@pancakes.eyeem.com/eyeem')
    connection = engine.connect()
except sqlalchemy.exc.SQLAlchemyError as e:
    engine = None
    connection = None
    print 'DB connection could not be established: {}'.format(repr(e))


def is_eyeem_photo_url(url):
    p = re.compile('https:\/\/www\.eyeem.com\/p\/\w+')
    if p.match(url) is None:
        return False
    else:
        return True


def is_eyeem_cdn_url(url):
    p = re.compile("http:\/\/cdn\.eyeem\.com\/thumb\/"
                   "(w\/\d*\/|)"
                   "("
                   "[\d\w]{40}"
                   "((-\d{10}|-\d{8})|)"
                   "(\/w\/\d*|)"
                   "|"
                   "[\d\w]{12}"
                   ")")
    # covers new and old versions:
    # http://cdn.eyeem.com/thumb/w/480/950f01de7ec28f3ddfaadade45dc0dab98106d59-1382579645 and
    # http://cdn.eyeem.com/thumb/w/480/8c5c5e58fae1e8ae087f5e39e017c2b14b8bc9d4-13777253 and
    # http://cdn.eyeem.com/thumb/w/1200/bymail4nkljl
    # http://cdn.eyeem.com/thumb/c0e745ddf389b21039ab221e96371f31823c4f1e-1423691544/w/480
    # ToDo: convert to list of multiple ifs for better readability, with warning if possible match
    if p.match(url) is None:
        return False
    else:
        return True


def is_eyeem_file_hash(file_hash_string):
    p = re.compile("(w\/\d*\/|)"
                   "("
                   "[\d\w]{40}"
                   "(-\d{10}|)"
                   "(\/w\/\d*|)"
                   "|"
                   "[\d\w]{12}"
                   ")")
    if p.match(file_hash_string) is None:
        return False
    else:
        return True


def eyeem_filename_to_cdn_url(filename, width=1024):
    cdn_url = 'http://cdn.eyeem.com/thumb/w/{}/'.format(width) + filename
    return cdn_url


def is_eyeem_url(url):
    file_name = url.split('/')[-1]
    return is_eyeem_file_hash(file_name)


def remove_non_eyeem_urls(urls):
    eyeem_urls = []
    for url in urls:
        if is_eyeem_url(url):
            eyeem_urls.append(url)
    return eyeem_urls


def lookup_file_hashes_from_photo_ids(photo_ids, verbose=True):
    _pre_checks(photo_ids)
    photo_ids = [str(entry) for entry in photo_ids]
    query = "select `filename`, eyeem_photo.id from eyeem_photo where eyeem_photo.id IN ({})".format(','.join(photo_ids))
    result_proxy = connection.execute(query)
    query_result = dict()
    for filename, photo_id in result_proxy:
        query_result[photo_id] = filename
    file_hash_mapping = dict()
    for photo_id in photo_ids:
        if long(photo_id) in query_result:
            if is_eyeem_file_hash(query_result[long(photo_id)]):
                file_hash_mapping[photo_id] = query_result[long(photo_id)]
            else:
                file_hash_mapping[photo_id] = ''
                if verbose:
                    print 'skipping no standard file hash {} for {}'.format(query_result[long(photo_id)], photo_id)
        else:
            file_hash_mapping[photo_id] = ''
            if verbose:
                print 'no filehash found for {}'.format(photo_id)
    return file_hash_mapping


def lookup_photo_ids_from_file_hashes(file_hashes, verbose=True):
    _pre_checks(file_hashes)
    query = "select eyeem_photo.id, eyeem_photo.filename from eyeem_photo where eyeem_photo.filename IN (\"{}\")".format('\",\"'.join(file_hashes))
    result_proxy = connection.execute(query)
    query_result = dict()
    for filename, file_hash in result_proxy:
        query_result[file_hash] = filename
    file_hash_mapping = dict()
    for file_hash in file_hashes:
        if file_hash in query_result:
            photo_id = query_result[file_hash]
            if is_photo_id(photo_id):
                file_hash_mapping[file_hash] = photo_id
            else:
                if verbose:
                    print 'skipping no standard photo_id {} for {}'.format(photo_id, file_hash)
        else:
            if verbose:
                print 'no photo_id found for {}'.format(file_hash)
    return file_hash_mapping


def get_cdn_path_from_photo_ids(photo_ids, w=1200):
    file_hash_mapping = lookup_file_hashes_from_photo_ids(photo_ids)
    file_hash_list = [entry for key, entry in file_hash_mapping.items()]
    cdn_urls = list()
    for file_hash in file_hash_list:
        if file_hash == '':
            url = ''
        else:
            url = 'http://cdn.eyeem.com/thumb/w/{}/{}'.format(w, file_hash)
        cdn_urls.append(url)
    cdn_urls_mapping = dict()
    for url, photo_id in zip(cdn_urls, file_hash_mapping):
        cdn_urls_mapping[photo_id] = url
    return cdn_urls_mapping


def is_photo_id(photo_id):
    if re.compile('\d{2,9}').match(str(photo_id)) is None:
        return False
    else:
        return True


def standardize_getty_tag(raw_getty_tag):
    return unicodedata.normalize('NFKD', raw_getty_tag.decode('latin-1')).lower().strip()


def lookup_keywords_from_photo_ids(photo_ids, return_raw=False):
    _pre_checks(photo_ids)
    photo_ids = [str(entry) for entry in photo_ids]
    query = "select `Original Filename`, `external keywords` from getty_photo " \
            "where `Original Filename` IN ({})".format(','.join(photo_ids))
    result_proxy = connection.execute(query)
    query_result = dict()
    for photo_id, raw_keywords in result_proxy:
        if raw_keywords is None:
            raw_keywords = ''
        if return_raw:
            query_result[photo_id] = raw_keywords.split(',')
        else:
            query_result[photo_id] = [standardize_getty_tag(entry) for entry in raw_keywords.split(',')]
    return query_result


def _pre_checks(input_data):
    if len(input_data) == 0:
        raise ValueError('list empty')
        # otherwise the code would raise an sql syntax exception
    if connection is None:
        raise EnvironmentError('Connection to DB not available')


def read_file_list(filename):

    try:
        pfile = open(filename, 'r')
        content = pfile.readlines()
        pfile.close()
        content = [c.strip() for c in content]
    except Exception as e:
        print e
        return

    return content


def download_image_and_save_as_jpeg(url, photo_id, download_path, fmt='PNG', time_out_image_downloading=3):
    response = requests.get(url, timeout=time_out_image_downloading)
    image = Image.open(StringIO.StringIO(response.content))
    image.load()
    if not os.path.exists(os.path.dirname(download_path)):
        os.makedirs(os.path.dirname(download_path))
    image.save(download_path, fmt)


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
