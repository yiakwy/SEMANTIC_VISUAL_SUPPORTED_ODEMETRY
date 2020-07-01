import cv2
import numpy as np

# we will use linear_assignment to quickly write experiments,
# later a customerized KM algorithms with various optimization in c++ is employed
# see https://github.com/berhane/LAP-solvers

# This is used for "Complete Matching" and we can remove unreasonable "workers" first and then apply it
import scipy.optimize as Optimizer

# This is used for "Maximum Matching". There is a desired algorithm implementation for our references
import scipy.sparse.csgraph as Graph

from svso.lib.maths.nputil import IoU_numeric, UIoU_numeric, cosine_dist
from svso.config import Settings

settings = Settings()

# setting debug variable
DEBUG = settings.DEBUG

# Linear Assignment Problems Solver Wrapper
class ROIMatcher:
    from enum import Enum
    class Algorithm(Enum):
        COMPLETE_MATCHING = 0
        MAXIMUM_MATCHING = 1

    def __init__(self):
        self.algorithm = ROIMatcher.Algorithm.COMPLETE_MATCHING
        self.THR = 0.85 # 0.75
        pass

    def mtch(self, trackList, detected_objects, product="composite"):
        N = len(trackList)
        M = len(detected_objects)

        weights = np.zeros((N, M))

        distance = np.zeros((N, M))
        corr = np.zeros((N, M))

        def make_standard_tf_box(box):
            y1, x1, y2, x2 = box
            return np.array([x1, y1, x2, y2])

        def compose_feat_vec(roi_feats, encodedId, score):
            new_feats = np.concatenate([roi_feats, encodedId, np.array([score])], axis=0)
            return new_feats

        INF = float("inf")
        EPILON = 1e-9

        column_names = list(map(lambda detection: str(detection), detected_objects))
        row_names = list(map(lambda landmark: str(landmark), trackList))

        for i in range(N):
            for j in range(M):
                obj1 = trackList[i]
                obj2 = detected_objects[j]

                # deep feature score
                ext_feat1 = compose_feat_vec(obj1.roi_features['roi_feature'],
                                             obj1.roi_features['class_id'],
                                             obj1.roi_features['score'])
                ext_feat2 = compose_feat_vec(obj2.roi_features['roi_feature'],
                                             obj2.roi_features['class_id'],
                                             obj2.roi_features['score'])

                # compute cosine distance
                score = cosine_dist(ext_feat1, ext_feat2)
                if np.isinf(score) or np.isnan(score):
                    raise Exception("Wrong Value!")
                corr[i, j] = score

                # must hold same semantic meaning if we belive our detectron
                if obj1.roi_features['label'] != obj2.roi_features['label']:
                    weights[i, j] = 1000
                    continue

                box1 = obj1.predicted_states
                box2 = make_standard_tf_box(obj2.projected_pos)

                left_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                right_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

                # 0 ~ 1
                # iou = IoU_numeric(box1, box2, left_area, right_area)
                # distance[i,j] = 1. - iou

                uiou = UIoU_numeric(box1, box2, left_area, right_area)
                distance[i, j] = (1 + uiou) / 2.0
                if np.isinf(uiou) or np.isnan(uiou):
                    raise Exception("Wrong Value!")

                # assign IoU distance
                # weights[i,j] = 1. - iou

                # assign UIoU distance
                if product == "composite":
                    weights[i, j] = (1 + uiou) / 2.0
                    # compute total score
                    weights[i, j] *= score
                    weights[i, j] = 1 - weights[i, j]
                elif product == "feature_only":
                    weights[i, j] = 1 - score
                else:
                    raise Exception("Not Implemented Yet!")

        mtched, unmtched_landmarks, unmtched_detections = ([], [], [])
        row_indice, col_indice = [], []
        np.set_printoptions(precision=3)
        if self.algorithm is ROIMatcher.Algorithm.COMPLETE_MATCHING:
            if DEBUG:
                # print weight matrix
                # print("%d landmarks, %d detections, forms %d x %d cost matrix :" % (N, M, N, M))
                # print(weights)
                pass

            # remove rows if there are no reasonable matches from cols so that we could
            # apply maximum match here. I have to say that this is very important!

            # @todo : TODO

            # see http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html,
            # also see https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/120020
            try:
                row_indice, col_indice = Optimizer.linear_sum_assignment(weights)
            except Exception as e:
                print(e)
                import pandas as pd
                # iou scores
                df = pd.DataFrame(distance, index=row_names, columns=column_names)
                # print("UIoUs:")
                # print(df)

                # entropy scores
                df = pd.DataFrame(corr, index=row_names, columns=column_names)
                # print("Corr:")
                # print(df)
                raise (e)


        else:
            raise Exception("Not Implemented Yet!")

        # use maximum matching strategy
        assignment = np.zeros((N, M))
        for i, col in enumerate(col_indice):
            row = row_indice[i]
            if weights[row, col] > self.THR:
                unmtched_landmarks.append(row)
                unmtched_detections.append(col)
                continue
            mtched.append((row, col, weights[row, col], distance[row, col]))
            assignment[row, col] = 1

        for i in range(N):
            if i not in row_indice:
                unmtched_landmarks.append(i)

        for j in range(M):
            if j not in col_indice:
                unmtched_detections.append(j)

        import pandas as pd
        # iou scores
        df = pd.DataFrame(distance, index=row_names, columns=column_names)
        # print("UIoUs:")
        # print(df)

        # entropy scores
        df = pd.DataFrame(corr, index=row_names, columns=column_names)
        # print("Corr:")
        # print(df)

        # draw matches

        df = pd.DataFrame(np.array(assignment), index=row_names, columns=column_names)
        # print("assignment:")
        # print(df)

        return mtched, unmtched_landmarks, unmtched_detections