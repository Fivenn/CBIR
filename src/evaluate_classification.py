# -*- coding: utf-8 -*-

from __future__ import print_function

from scipy import spatial
import numpy as np
import pandas as pd


class EvaluateClassification(object):

    def make_samples(self):
        raise NotImplementedError("Needs to implemented this method")


def distance(v1, v2, d_type='d1'):
    ''' Calculates the distance between two vectors according to a distance type.
        arguments
            v1     : The first vector used to calculate distance
            v2     : The second vector used to calculate distance
            d_type : The type of distance to be calculated

        return
            The distance between two vectors, depending on the type of distance selected

    '''
    assert v1.shape == v2.shape, "shape of two vectors need to be same!"

    if d_type == 'd1':
        return np.sum(np.absolute(v1 - v2))
    elif d_type == 'd2':
        return np.sum((v1 - v2) ** 2)
    elif d_type == 'd2-norm':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'd3':
        pass
    elif d_type == 'd4':
        pass
    elif d_type == 'd5':
        pass
    elif d_type == 'd6':
        pass
    elif d_type == 'd7':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'd8':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'cosine':
        return spatial.distance.cosine(v1, v2)
    elif d_type == 'square':
        return np.sum((v1 - v2) ** 2)


def weightDistance(results):
    ''' Calculate a weight corresponding to the calculation of the average distance for each class.
    argument
        results : The distances between a query and samples

    return
        averageClassDistance : A weight corresponding to the calculation of the average distance for each class.
    '''
    resultsDataFrameGroups = pd.DataFrame(results).groupby("cls")
    averageClassDistance = []

    for name, group in resultsDataFrameGroups:
        averageClassDistance.append({'cls': name, 'averageClassDistance': np.mean(group['dis'])})

    return averageClassDistance


def infer(query, samples=None, db=None, sample_db_fn=None, depth=None, d_type='d1'):
    ''' Infer a query and return its weighted distance
        arguments
            query        : A dictionnary with 3 keys
                {
                    img  : path of image,
                    cls  : class of image
                    hist : histogram of image
                }
            samples      : A list of dictionnary
                {
                    img  : path of image,
                    cls  : class of image
                    hist : histogram of image
                }
            db           : An instance of DB class
            sample_db_fn : A function making samples from DB instance 
            depth        : Retrieved depth during inference, by default, it is equal to DB size
            dtype        : The type of distance to be calculated

        return
            weightedDistance : The weighted distance of a query infered from the samples
    '''
    assert samples != None or (
            db != None and sample_db_fn != None), "need to give either samples or db plus sample_db_fn"
    if db:
        samples = sample_db_fn(db)

    q_img, q_hist = query['img'], query['hist']
    results = []
    for _, sample in enumerate(samples):
        s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
        if q_img == s_img:
            continue
        results.append({
            'dis': distance(q_hist, s_hist, d_type=d_type),
            'cls': s_cls
        })
    results = sorted(results, key=lambda x: x['dis'])
    if depth and depth <= len(results):
        results = results[:depth]
    weightedDistance = sorted(weightDistance(results), key=lambda x: x['averageClassDistance'])

    return weightedDistance


def evaluate_class(db, f_class=None, f_instance=None, depth=None, d_type='d1'):
    ''' Evaluates each image in the database in order to infer a class according to the chosen feature.
        arguments
            db         : an instance of DB class
            f_class    : A class that generate features, used to generate samples
            f_instance : A instance that generate features, used to generate samples
            depth      : Retrieved depth during inference, by default, it is equal to DB size
            d_type     : distance type to be calculated

        returns
            ok           : The number of classes classified according to a feature
            len(samples) : The number of samples processed
    '''
    assert f_class or f_instance, "needs to give class_name or an instance of class"
    
    if f_class:
        f = f_class()
    elif f_instance:
        f = f_instance
    samples = f.make_samples(db)
    ok = 0
    for query in samples:
        result = infer(query, samples=samples, depth=depth, d_type=d_type)
        if (query['cls'] == result[0]['cls']):
            ok += 1

    return ok, len(samples)
