import cv2
import numpy as np
from pysvso.config import Settings

settings = Settings()

# setting debug variable
DEBUG = settings.DEBUG

class OpticalFlowBasedKeyPointsMatcher:

    def __init__(self):
        self.kps = None
        self.features = None
        # changed from 7 -> 14, in case that calcOpticalFlowFarneback algorithm gives wrong prediction results
        self.R = 12  # px
        self.sample_size = -1
        self.matches = None
        self.mask = None
        self.THR = 0.75

    def Init(self):
        return self

    def set_FromFrame(self, frame):
        self._frame = frame
        return self

    def set_FromKP(self, kps):
        self.kps = kps
        return self

    def set_FromFeatures(self, features):
        self.features = features
        return self

    # @todo : TODO impl
    def _get_neighbors(self, row, col, feat_map, img_shp):
        H, W = img_shp[0:2]
        x1, y1 = (col - self.R, row - self.R)
        x2, y2 = (col + self.R, row + self.R)

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0

        if x2 < 0:
            x2 = 0
        if y2 < 0:
            y2 = 0

        if x2 >= W:
            x2 = W - 1
        if y2 >= H:
            y2 = H - 1

        indice = feat_map[y1:y2, x1:x2] != -1
        return feat_map[y1:y2, x1:x2][indice]

    # @todo : TODO impl
    def mtch(self, kps, other_features, last_frame):
        mtches = []
        img_shp = self._frame.img.shape[0:2]

        # predict coord
        kps_coor_r = list(map(lambda kp: kp.pt, kps))
        predicted_pos = last_frame.predictors["OpticalFlowKPnt"].predict(kps_coor_r, self._frame.img_grey())

        # init feat_map
        feat_map = np.full(img_shp, -1)
        for i, kp in enumerate(predicted_pos):
            x, y = kp
            if int(y) >= img_shp[0] or int(y) < 0 or \
                    int(x) >= img_shp[1] or int(x) < 0:
                continue
            feat_map[int(y), int(x)] = i

        def _hamming_distance(x, y):
            from scipy.spatial import distance
            return distance.hamming(x, y)

        #
        for i, kp in enumerate(self.kps):
            x, y = kp.pt
            indice = self._get_neighbors(int(y), int(x), feat_map, img_shp)
            if len(indice) == 0:
                continue

            # KNN search
            feat_l = self.features[i]

            dist = None
            min_dist, min_ind = np.inf, None

            for ind in indice:
                feat_r = other_features[ind]
                dist = _hamming_distance(feat_l, feat_r)
                if min_dist > dist:
                    min_dist = dist
                    min_ind = ind
            try:
                if min_dist > self.THR:
                    continue
                mtches.append(cv2.DMatch(i, min_ind, min_dist))
                if DEBUG:
                    kpl = kp
                    kpr = kps[min_ind]
                    if np.sqrt(np.power(kpl.pt[0] - kpr.pt[0], 2) + \
                               np.power(kpr.pt[1] - kpr.pt[1], 2)) > self.R:
                        pass
                        # raise Exception("Wrong Match!")

            except Exception as e:
                print(e)
                print("i", i)
                print("kpl(cur)", kp.pt)
                print("min_ind", min_ind)
                print("kpr(last_frame)", kps[min_ind].pt)
                print("predicted kpr(last_frame)", predicted_pos[min_ind])
                print("min_dist", min_dist)
                print("dist", dist)
                raise (e)

        # sort keypoints
        mtches = sorted(mtches, key=lambda mtch: mtch.distance)

        # filter out distance larger than 0.8
        # mtches = list(filter(lambda mtch: mtch.distance < 0.8, mtches))

        import pandas as pd

        distances = [mtch.distance for mtch in mtches]

        df = pd.DataFrame({
            "Dist": distances
        })

        print("[OpticalFlowBasedKeyPointsMatcher] matched distances (hamming distance):\n")
        print(df)

        l = len(self.kps)
        mask = np.ones((l, 1))

        #
        self.matches = mtches
        self.mask = mask

        return mtches, mask.tolist()


# using cv2.FlannBasedMatcher as backend to match key points
class FlannBasedKeyPointsMatcher:

    def __init__(self):
        self.features = None
        self._impl = None
        self.LOWE_RATIO_TEST_ON = True
        self.sample_size = -1

    def Init(self):
        # FLANN parameters, simply borrow from opencv website: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        # use cv2 implementation as backend
        self._impl = cv2.FlannBasedMatcher(index_params, search_params)
        self.K = 2
        return self

    def set_FromFrame(self, frame):
        self._frame = frame
        return self

    def set_FromFeatures(self, features):
        self.features = features
        return self

    # @todo : TODO impl
    def mtch(self, other_features):
        """
        @return mtched : Tuple(List<Tuple(DMatch, DMatch)>, list (2d))
        """

        # opencv2 flann matcher for more information, also see this
        # https://answers.opencv.org/question/192712/why-does-knnmatch-return-a-list-of-tuples-instead-of-a-list-of-dmatch/
        mtches = self._impl.knnMatch(np.asarray(self.features, np.float32),
                                     np.asarray(other_features, np.float32),
                                     self.K)

        # filtered matches
        # @todo : TODO impl
        l = len(mtches)
        mask = np.ones((l, self.K))

        def ratio_test(mask, mtches):
            # ratio test as per Lowe's paper
            for i, row in enumerate(mtches):
                if row[0].distance >= 0.7 * row[1].distance:
                    # always ignore the second match object
                    mask[i, 0] = 0

        # Lowe ratio test
        if self.LOWE_RATIO_TEST_ON:
            ratio_test(mask, mtches)

        if self.sample_size is not -1:
            print("Filtering out samples ...")
            choosed_indice_mask = np.where(mask[:, 0] == 1)[0]
            permutated_choosed_indice_mask = np.random.permutation(choosed_indice_mask)

            # mask[permutated_choosed_indice_mask[self.sample_size:], 0] = 0
            indice = permutated_choosed_indice_mask[:self.sample_size]
            mtches = list(map(lambda idx: mtches[idx], indice))
            mask = mask[indice]
        else:
            print("Keep all key piont matches.")

        return mtches, mask.tolist()