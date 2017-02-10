#
# Author : Harsimrat Sadhawalia
# Custom loss functions for keras models
# 1. Mutlilabel Multiclass hinge loss
# 2. Others soon to follow
#

import numpy
import numpy as np

import theano
from theano import tensor as T
from theano import function

from keras.objectives import *
from keras import backend as K

# from keras.backend.common import _EPSILON
_EPSILON = 10e-8

# Hope no image would have more than 10000 labels in GT
weights = np.cumsum(1.0 / np.array(range(1, 10000 + 1)).reshape(1, -1))
warp_weight_func = theano.shared(weights)

# NDCG weights array rank


class ObjectiveFunctions():

    def __init__(self):

        self.weights = 1.0

        self.dict = {'multilabel_hinge': self.loss_multilabel_multiclass_hinge,
                        'pairwise_rank': self.loss_pairwise_ranking,
                        'hamming_loss': self.loss_hamming_similarity,
                        'warp': self.loss_warp,
                        'multilabel_rank': self.loss_multilabel_multiclass_ranking,
                        'categorical_crossentropy_pn': self.categorical_crossentropy_penalty,
                        'categorical_crossentropy_weighted': self.categorical_crossentropy_weighted,
                        'categorical_crossentropy_eyeem': self.categorical_crossentropy_eyeem,
                        'categorical_crossentropy': categorical_crossentropy,
                        'binary_crossentropy': binary_crossentropy,
                        'hinge': hinge,
                        'sq_hinge': squared_hinge,
                        'mse': mse,
                        'kl_divergence': self.kullback_leibler_divergence,
                        'fish': self.poisson,
                        'cosine': self.cosine_proximity,
                        'categorical_crossentropy_captions': self.categorical_crossentropy_captions,
                        'categorical_crossentropy_pn_captions': self.categorical_crossentropy_pn_captions}

    def set_weights(self, w):

        self.weights = w

    def categorical_crossentropy_penalty(self, y_true, y_pred):

        """
            Inputs : Expected 2D tensor for targets and predictions
            1. Target k-hot n_samples x n_classes
            2. Predictions in [0,1] n_samples x n_classes
            3. Chill or Panic depending on your results
        """

        # normalization
        y_pred /= y_pred.sum(axis=-1, keepdims=True)

        # avoid numerical instability with _EPSILON clipping
        y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

        ccp = - T.sum(y_true * T.log(y_pred) + (1 - y_true) * T.log(1 - y_pred), axis=-1)

        return ccp

    def categorical_crossentropy_eyeem(self, y_true, y_pred):

        """
            Inputs : Expected 2D tensor for targets and predictions
            1. Target k-hot n_samples x n_classes
            2. Predictions in [0,1] n_samples x n_classes
            3. Chill or Panic depending on your results
        """

        # normalization
        y_pred /= y_pred.sum(axis=-1, keepdims=True)

        # avoid numerical instability with _EPSILON clipping

        y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

        cce = -T.sum(y_true * T.log(y_pred), axis=- 1)

        return cce

    def categorical_crossentropy_weighted(self, y_true, y_pred):

        """
            Inputs : Expected 2D tensor for targets and predictions
            1. Target k-hot n_samples x n_classes
            2. Predictions in [0,1] n_samples x n_classes
            3. Chill or Panic depending on your results
        """

        # normalization
        y_pred /= y_pred.sum(axis=-1, keepdims=True)

        # avoid numerical instability with _EPSILON clipping
        y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

        ccew = -T.sum(self.weights * y_true * T.log(y_pred), axis=y_true.ndim - 1)

        return ccew

    def loss_multilabel_multiclass_hinge(self, y_true, y_pred, margin=1):

        """
            Inputs : Expected 2D tensor for targets and predictions
            1. Target k-hot n_samples x n_classes
            2. Predictions in [0,1] n_samples x n_classes
            3. To use this you need to take replace sigmoid or softmax with linear at the last layer while training
            4. Put back softmax or sigmoid back for testing
            5. Chill or Panic depending on your results
        """

        num_samples = y_true.shape[0]
        num_classes = y_true.shape[1]

        def sample_loop(s, margin, y_true, y_pred):

            true, pred = y_true[s], y_pred[s]

            pos = pred[true.nonzero()]
            neg = pred[(1 - true).nonzero()]

            pos_tile = T.extra_ops.repeat(pos, neg.shape[0]).reshape((pos.shape[0], neg.shape[0]))
            neg_tile = T.extra_ops.repeat(neg, pos.shape[0]).reshape((neg.shape[0], pos.shape[0])).transpose()

            p_loss = margin + neg_tile - pos_tile

            loss = p_loss[T.nonzero(p_loss > 0)]

            return T.sum(loss)

        mlh, _ = theano.scan(fn=sample_loop, outputs_info=None, sequences=T.arange(num_samples), non_sequences=[margin, y_true, y_pred])

        return T.mean(mlh)

    def categorical_crossentropy_captions(self, y_true, y_pred):

        """
            Inputs : Expected 3D tensor for targets and predictions
            1. Target 1-hot n_samples x time_steps x vocab_size
            2. Predictions in [0,1] n_samples x time_steps x vocab_size
            3. Chill or Panic depending on your results
        """

        # normalization
        y_pred /= y_pred.sum(axis=-1, keepdims=True)

        # avoid numerical instability with _EPSILON clipping
        y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

        ccc = - T.sum(T.sum(y_true * T.log(y_pred), axis=-1), axis=-1)

        return ccc

    def categorical_crossentropy_pn_captions(self, y_true, y_pred):

        """
            Inputs : Expected 3D tensor for targets and predictions
            1. Target 1-hot n_samples x time_steps x vocab_size
            2. Predictions in [0,1] n_samples x time_steps x n_classes
            3. Chill or Panic depending on your results
        """

        # normalization
        y_pred /= y_pred.sum(axis=-1, keepdims=True)

        # avoid numerical instability with _EPSILON clipping
        y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

        ccp = - T.sum(T.sum(y_true * T.log(y_pred) + (1 - y_true) * T.log(1 - y_pred), axis=-1), axis=-1)

        return ccp

    def loss_pairwise_ranking(self, y_true, y_pred):

        """
            Inputs : Expected 2D tensor for targets and predictions
            1. Target k-hot n_samples x n_classes
            2. Predictions in [0,1] n_samples x n_classes
            3. Take out sigmoid activation on the last layer
            4. Put back softmax or sigmoid back for testing
            5. Chill or Panic depending on your results
        """

        num_samples = y_true.shape[0]
        num_classes = y_true.shape[1]

        def sample_loop(s, y_true, y_pred):

            true, pred = y_true[s], y_pred[s]

            pos = pred[true.nonzero()]
            neg = pred[(1 - true).nonzero()]

            pos_tile = T.extra_ops.repeat(pos, neg.shape[0]).reshape((pos.shape[0], neg.shape[0]))
            neg_tile = T.extra_ops.repeat(neg, pos.shape[0]).reshape((neg.shape[0], pos.shape[0])).transpose()

            p_loss = 1 + neg_tile - pos_tile

            loss = p_loss[T.nonzero(p_loss > 0)]

            return T.sum(loss)

        pwr, _ = theano.scan(fn=sample_loop, outputs_info=None, sequences=T.arange(num_samples), non_sequences=[y_true, y_pred])

        return T.mean(pwr)

    def loss_hamming_similarity(self, y_true, y_pred, list_size=4, margin=1, alpha=1):

        """
            Inputs : Expected 2D tensor for targets and quadruplets for prediction
            1. Predictions n_samples x n_bits
            2. Target as quadruplets relevance rank [r_0,r_1,r_2,r_3]
            3. Have sigmoid as activation in the last layer
            4. Chill or Panic depending on your results
        """

        # 0. Scale y_pred by 2 and subtract 1, for getting bits in [-1,+1]
        # 1. Compute NDCG weights for rank list
        # 2. Compute hamming distance for rank list and add margin
        # 3. Take dot product of 1. and 2. eq.4
        # 4. Compute mean of each bit over the batch average to 0
        # 5. Add L2 reglurization to hash layer net loss as eqn 6 [ Handled by the keras layer regulalization]
        # 6. Net loss is 3. + 4. + 5.

        num_samples = y_true.shape[0]
        n_queries = y_true.shape[0] // list_size
        K = y_pred.shape[1]

        y_pred = 2 * y_pred - 1

        def hammming_distance_loss(s, y_pred, p_weights, K, margin):

            def hammming_distance(d, q, K):
                return (K - T.dot(q, d)) // 2.0

            q = y_pred[s]

            d_0 = y_pred[s + 1]
            d_1 = y_pred[s + 2]
            d_2 = y_pred[s + 3]

            w_0 = y_true[s + 1]
            w_1 = y_true[s + 2]
            w_2 = y_true[s + 3]

            p_dist = theano.scan(fn=hammming_distance, outputs_info=None, sequences=[d_0, d_1, d_2], non_sequences=[q, K])
            loss = (w_0 - w_1) * T.max(0, p_dist[1] - p_dist[0] + margin) + (w_0 - w_2) * T.max(0, p_dist[2] - p_dist[0] + margin) + (w_1 - w_2) * T.max(0, p_dist[2] - p_dist[1] + margin)

            return loss

        def rank_discount(s, y_true):
            return T.pow(2, y_true[s])

        p_weights = theano.scan(fn=rank_discount, outputs_info=None, sequences=T.arange(num_samples), non_sequences=[y_true])
        p_loss, _ = theano.scan(fn=hammming_distance_loss, outputs_info=None, sequences=T.arange(0, num_samples, list_size), non_sequences=[y_pred, p_weights, K, margin])

        mean_bits = T.mean(y_pred, axis=0)
        mean_bits_loss = alpha * T.dot(mean_bits, mean_bits)

        hamming_loss = T.mean(p_loss, axis=-1)

        return hamming_loss + mean_bits_loss

    def loss_warp(self, y_true, y_pred):

        """
            Inputs : Expected 2D tensor for targets and predictions
            1. Target k-hot n_samples x n_classes
            2. Predictions in [0,1] n_samples x n_classes
            3. Take out sigmoid activation on the last layer
            4. Put back softmax or sigmoid back for testing
            5. Chill or Panic depending on your results
        """

        num_samples = y_true.shape[0]
        num_classes = y_true.shape[1]

        def trials(s, condition, n_trials):

            r = np.random.random_integers(0, n_trials)

            return condition[r] > 0

        def rank_weights(n_pos, p_loss):

            n_trials = p_loss.shape[1] // 2

            condition = p_loss[n_pos]

            s = theano.scan(fn=trials, outputs_info=None, sequences=T.arange(n_trials), non_sequences=[condition, n_trials])

            return 0 if s.nonzero()[0].shape[0] == 0 else warp_weight_func[s.nonzero()[0][0]]

        def sample_loop(s, y_true, y_pred):

            true, pred = y_true[s], y_pred[s]

            pos = pred[true.nonzero()]
            neg = pred[(1 - true).nonzero()]

            pos_tile = T.extra_ops.repeat(pos, neg.shape[0]).reshape((pos.shape[0], neg.shape[0]))
            neg_tile = T.extra_ops.repeat(neg, pos.shape[0]).reshape((neg.shape[0], pos.shape[0])).transpose()

            p_loss = 1 + neg_tile - pos_tile

            # WARP loss function
            l_rank = theano.scan(fn=rank_weights, outputs_info=None, sequences=T.arange(p_loss.shape[0]), non_sequences=[p_loss])

            l_rank = l_rank.reshape([1, -1]).transpose()

            p_loss *= l_rank

            return T.sum(p_loss, axis=-1)

        warp, _ = theano.scan(fn=sample_loop, outputs_info=None, sequences=T.arange(num_samples), non_sequences=[y_true, y_pred])

        return T.mean(warp)

    def loss_multilabel_multiclass_ranking(self, y_true, y_pred):

        """
            Inputs : Expected 2D tensor for targets and predictions
            1. Target k-hot n_samples x n_classes
            2. Predictions in [0,1] n_samples x n_classes
            3. Have sigmoid as activation in the last layer
            4. Chill or Panic depending on your results
        """

        num_samples = y_true.shape[0]
        num_classes = y_true.shape[1]

        # make a ranking weight matrix
        ranking_weights = T.tile(t_weights, (num_samples, 1))

        # sort score ( descending-order )
        # cant use sort non diffentiable replace with something approximate
        class_ranks = T.argsort(-y_pred, axis=1)

        # take labels of top predictions [ Mind ducking indexing here, brace yourself ]
        class_order = y_true[T.range(num_samples)[:, None], class_ranks]

        # negation of the results
        p_loss_true = 1 - class_order

        # take loss of high ranking incorrect predictions
        p_loss_true_mlr = ranking_weights[p_loss_true.nonzero()]

        mlr = T.mean(p_loss_true_mlr, axis=-1)

        return mlr

    def kullback_leibler_divergence(self, y_true, y_pred):

        y_true = K.clip(y_true, _EPSILON, 1)
        y_pred = K.clip(y_pred, _EPSILON, 1)

        return T.sum(y_true * K.log(y_true / y_pred), axis=-1)

    def poisson(self, y_true, y_pred):

        return T.mean(y_pred - y_true * T.log(y_pred + _EPSILON()), axis=-1)

    def cosine_proximity(self, y_true, y_pred):

        y_true = K.l2_normalize(y_true, axis=-1)
        y_pred = K.l2_normalize(y_pred, axis=-1)

        return 1 - K.sum(y_true * y_pred, axis=-1)
