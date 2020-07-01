import numpy as np
import cv2
import logging
_logger = logging.getLogger("predictor.opticalflow")

from svso.log import LoggerAdaptor


# According to partial equations of images movement, we have
#   uf_x + vf_y + f_t = 0
# u and v are unknow variables while (f_x, f_y) are computed numeric gradients of an image. This is a linear equation. Hence
# we can sample gradients to estimate the movement of images. This gives a strong estimate movement
# of detected objects in pixel level.
# Note, this is a different from primitive KalmanFilter which applies physics assumptions to
# projected objects onto images.
#
# The algorithm only apply to grey level images.
#
# @todo : TODO impl
class Grid(object):
    pass


class OpticalFlowBBoxPredictor(object):
    logger = LoggerAdaptor("OpticalFlowBBoxPredictor", _logger);

    def __init__(self):
        #
        self._impl = None

        #
        self._cur_img = None

        #
        self._pre_img = None

        #
        self.IMAGE_SHAPE = (None, None)

        # dense optical flow for the image
        self._flow = None

    def Init(self):
        # I am also considering to use createOptFlow_DualTVL1 algoithm ...
        # Farneback algorithm has some problem in numeric computing : https://stackoverflow.com/questions/46521885/why-does-cv2-calcopticalflowfarneback-fail-on-simple-synthetic-examples
        self._impl = cv2.calcOpticalFlowFarneback
        return self

    def set_FromImg(self, img):
        self._cur_img = img
        self.IMAGE_SHAPE = img.shape[0:2]
        return self

    def get_flow(self):
        return self._flow

    def set_flow(self, flow):
        self._flow = flow
        return self

    def predict(self, states, observed_img=None):
        assert (self._cur_img is not None)
        flow = self._flow
        if flow is None:
            assert (observed_img is not None)
            params = {
                'pyr_scale': 0.8,
                'levels': 3,
                'winsize': 31,
                'iterations': 5,
                'poly_n': 5,
                'poly_sigma': 1.2,
                'flags': 0
            }
            self._pre_img = self._cur_img
            self._flow = self._impl(self._pre_img, observed_img, None,
                                    params['pyr_scale'],
                                    params['levels'],
                                    params['winsize'],
                                    params['iterations'],
                                    params['poly_n'],
                                    params['poly_sigma'],
                                    params['flags'])
            self._cur_img = observed_img
            flow = self._flow

        dx, dy = flow[:, :, 0], flow[:, :, 1]
        x1, y1, w, h = states
        x2, y2 = x1 + w, y1 + h

        H, W = self.IMAGE_SHAPE
        if x2 >= W:
            x2 = W - 1;
        if y2 >= H:
            y2 = H - 1;

        ret = np.array(
            [x1 + dx[int(y1), int(x1)], y1 + dy[int(y1), int(x1)], w + dx[int(y2), int(x2)] - dx[int(y1), int(x1)],
             h + dy[int(y2), int(x2)] - dy[int(y1), int(x1)]])
        return ret

    def update(self, measures):
        import sys
        func_name = sys._getframe().f_code.co_name
        raise NotImplemented("the method %s is not implemented!" % func_name)


# @todo : TODO using goodFeaturesToTrack to improve dense estimation of flow
class OpticalFlowKPntPredictor(OpticalFlowBBoxPredictor):
    logger = LoggerAdaptor("OpticalFlowKPntPredictor", _logger);

    def __init__(self):
        super().__init__()

    def predict(self, kps, observed_img=None):
        assert (self._cur_img is not None)
        flow = self._flow
        if flow is None:
            assert (observed_img is not None)
            params = {
                'pyr_scale': 0.8,
                'levels': 3,
                'winsize': 51,
                'iterations': 5,
                'poly_n': 7,
                'poly_sigma': 1.2,
                'flags': 0
            }
            self._pre_img = self._cur_img
            self._flow = self._impl(self._pre_img, observed_img, None,
                                    params['pyr_scale'],
                                    params['levels'],
                                    params['winsize'],
                                    params['iterations'],
                                    params['poly_n'],
                                    params['poly_sigma'],
                                    params['flags'])
            self._cur_img = observed_img
            flow = self._flow

        # see help lib, adapted from opencv4_source_code/samples/python/opt_flow.py
        # about how to use `flow` object
        dx, dy = flow[:, :, 0], flow[:, :, 1]

        l = len(kps)
        new_pixels = np.zeros((l, 2))
        # @todo : TODO impl
        try:
            for i, kp in enumerate(kps):
                x1, y1 = kp
                x2, y2 = x1 + dx[int(y1), int(x1)], y1 + dy[int(y1), int(x1)]
                new_pixels[i, 0] = x2
                new_pixels[i, 1] = y2
        except Exception as e:
            print(e)
            print("kp", kp)
            raise (e)

        return new_pixels


# @todo : TODO impl
class HybridOpticalFlowFilter(object):

    def __init__(self):
        pass
