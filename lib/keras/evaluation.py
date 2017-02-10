import numpy
from sklearn.metrics import precision_recall_curve, f1_score
   
class StatsError(Exception):
    pass


def recall_at_specified_precision(labels, predictions, specified_precision=0.95):
    p, r, t = precision_recall_curve(labels, predictions)
    index = numpy.where(p >= specified_precision)[0][0]
    return r[index]


def threshold_at_specified_precision(labels, predictions, specified_precision=0.95, raise_if_not_met=False, legacy=False):
    """Predicted scores need to be higher or equal to the returned threshold to reach the specified precision"""
    if max(predictions) > 1.0:
        raise ValueError("max(predictions) is higher than 1.0".format(max(predictions)))
    p, r, t = precision_recall_curve(labels, predictions)
    if legacy:
        index = numpy.where(p >= specified_precision)[0][0] - 1  # original wrong comment: "length of threshold is p-1"
        if index == -1:
            if p[0] >= specified_precision:
                index = 0
            elif raise_if_not_met is False:
                index = 0
            else:
                raise StatsError('Precision of {} is never reached. Max is {}'.format(specified_precision, p[0]))
        threshold = t[index]
        return threshold

    else:
        index = numpy.where(p >= specified_precision)[0][0]
    try:
        threshold = t[index]
    except IndexError:
        if raise_if_not_met:
            raise StatsError('Precision of {} is never reached. Max is {}'.format(specified_precision, p[0]))
        else:
            threshold = 1.0
    return threshold


def precision_for_specified_recall(labels, predictions, specified_recall=0.95):
    p, r, t = precision_recall_curve(labels, predictions)
    index = numpy.where(r >= specified_recall)[0][-1]
    if index == -1:
        raise ValueError
    return p[index]

def get_thresholds(predictions, complete_list_concepts, gt, PRECISION_THRESHOLD = 0.9, minimum_detections=1):
    thresholds = dict()
    for i, concept in enumerate(complete_list_concepts):
        true_labels = list()
        predicted_scores = list()
        list_photo_ids_predictions = set(predictions[concept].keys())
        list_photo_ids_gt = set(gt[concept])
        if len(list_photo_ids_predictions) == 0:
            thresholds[concept] = 1.0
            continue            
        for photo_id in list_photo_ids_predictions:
            score = predictions[concept][photo_id]
            true_labels.append(photo_id in list_photo_ids_gt),
            predicted_scores.append(score)
        threshold = threshold_at_specified_precision(true_labels,
                                            predicted_scores, 
                                        PRECISION_THRESHOLD)
        if len([entry for entry in predicted_scores if entry >= threshold]) < minimum_detections:
            threshold = 1.0
        thresholds[concept] = threshold 
    return thresholds

def get_detections_per_concept(predictions, complete_list_concepts, gt, thresholds):
    detections_per_concept = dict()
    for i, concept in enumerate(complete_list_concepts):
        true_labels = list()
        predicted_scores = list()
        list_photo_ids_predictions = set(predictions[concept].keys())
        list_photo_ids_gt = set(gt[concept])
        if len(list_photo_ids_predictions) == 0:
            detections_per_concept[concept] = 0
            continue            
        for photo_id in list_photo_ids_predictions:
            score = predictions[concept][photo_id]
            true_labels.append(photo_id in list_photo_ids_gt),
            predicted_scores.append(score)
        detections = numpy.sum(numpy.array(numpy.array(predicted_scores)>=thresholds[concept]))            
        detections_per_concept[concept] = detections 
    return detections_per_concept

def threshold_for_specified_recall(labels, predictions, specified_recall=0.95):
    p, r, t = precision_recall_curve(labels, predictions)
    index = numpy.where(r >= specified_recall)[0][-1]
    if index == -1:
        raise ValueError
    return t[index]


def top_k_precision(predicted_scores, labels, k=100):
    detections = numpy.argsort(predicted_scores, 0)[::-1][:k]
    predicted_precision_at_top_k = 0
    for index in detections:
        if labels[index] == 1:
            predicted_precision_at_top_k += 1
    return predicted_precision_at_top_k


def compute_accuracy(true_positives, false_positives):
    positives = true_positives + false_positives
    return 1.0 * true_positives / positives


def max_f1_score(true_labels, predicted_scores):
    result = max([f1_score(true_labels, predicted_scores > threshold)
                  for threshold in numpy.linspace(0, 1, 100)])
    return result


def max_f1_score_v2(true_labels, predicted_scores):
    lower_bound = min(predicted_scores)
    upper_bound = max(predicted_scores)
    result = max([f1_score(true_labels, predicted_scores > threshold)
                  for threshold in numpy.linspace(lower_bound, upper_bound, 100)])
    return result