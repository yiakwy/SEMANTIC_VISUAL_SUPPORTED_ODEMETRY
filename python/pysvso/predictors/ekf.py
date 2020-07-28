import numpy as np
import logging
_logger = logging.getLogger("predictor.kf")

from pysvso.lib.log import LoggerAdaptor

# This class implements universal Kalman Filter used by BBoxKalmanFilter
# BBoxKalmFilter give specific values setup for a general states predictor KalamFilter
# for video tracking mission.
#
# Author: Lei Wang
# Date: Feb 20, 2020
# reference of the implementation :
# 1. http://ros-developer.com/2019/04/11/extended-kalman-filter-explained-with-python-code/
# 2. Simple Online Realtime Tracking project.
#
# Credits to the relevant authors
class ExtendedKalmanFilter(object):
    logger = LoggerAdaptor("ExtendedKalmanFilter", _logger);

    def __init__(self, states_dim, measures_dim):
        """
        @param states_dim: int
        @param measures_dim: int
        """
        self.states_dim = states_dim
        self.measures_dim = measures_dim
        # states to inspect, change to symbol `x` during computation
        # timer series of states is maintained by a tracker with array of observation
        self._states = np.zeros((states_dim, 1))

        # usually I used 1/f where f estimated frequences of video sequences or observation sequences
        self.dt = 1.0 / 24
        # Constant Acceleration Physics Model, as long as dt is small enough, this equation holds
        self._A = self.populate_physics_constrain(self.dt)
        self._B = None
        self._u = None
        # See usage examples from https://github.com/balzer82/Kalman. However pay attention here that
        # my implementation is generalized to arbirary one-dimensional observations
        self.P = np.eye(states_dim)
        # observation variance matrix
        self.Q = self.populate_states_variance_constrain();
        # measurements cover matrix
        self.H = self.populate_measures_constrain();
        # variance matrix for measurements
        self.R = self.populate_priors_variance_constrain();

    def Init(self):
        pass

    ## helper func to initate Kalman Filter Computing routines.

    # The routines are also helpful when we implement multi fusion strategy for different sensors
    # since the frequencies vary dramatically, we could apply differnt measures when observation data from
    # sensors arrive.

    # @todo : TODO impl
    def populate_physics_constrain(self, dt):
        """
        @return A : np.array with shape of (states_dim, states_dim)
        """
        # for each observation x, we have
        # x_{k+1} = x_{k} + v_{k} * dt + 1.0/2 * a_{k+1}^2
        states_dim = self.states_dim

        A = np.eye(states_dim)
        factors = [dt, 0.5 * dt * dt]
        assert (self.states_dim % 3 == 0)
        first_order_observation_dim = self.states_dim / 3.0
        for i in range(states_dim):
            k = 1
            j = int(i + first_order_observation_dim * k)
            # print("i,j,k", i,j,k)
            while j < states_dim and k < len(factors):
                A[i, j] *= factors[k]
                k += 1
        return A

    # @todo : TODO impl
    def populate_measures_constrain(self, measures_indice=None):
        """
        @return H : np.array with shape of (measures_dim, states_dim)
        """
        H = np.zeros((self.measures_dim, self.states_dim))
        if measures_indice is not None and isinstance(measures_indice, (list, tuple)):
            assert (len(measures_indice) == self.measures_dim)
            # update H
            for i, idx in enumerate(measures_indice):
                H[i][idx] = 1
        else:
            if measures_indice is None:
                logging.warning(
                    "The measures indice is none. please set it using 'Kalmanfilter.populate_measures_constrain(self, indice)' later.")
            else:
                raise Exception(
                    "expect list or tuple, but encounter %s for measures_indice" % str(type(measures_indice)))
        return H

    def populate_priors_variance_constrain(self, measures_indice=None, measures_noises=None):
        """
        @return R : np.array with shape of (measures_dim, measures_dim)
        """
        R = np.eye(self.measures_dim)
        if measures_indice is not None and isinstance(measures_indice, (list, tuple)):
            if measures_noises is None:
                measures_noises = np.ones((self.measures_dim, 1))
            elif hasattr(measures_noises, "__len__"):
                # implements list interface
                pass
            else:
                raise Exception(
                    "expect `measures_noise` to be array alike object, but encounter %s" % str(type(measures_noise)))
            assert (len(measures_indice) == self.measures_dim)
            # update R
            R = np.diag(measures_noises)
        else:
            if measures_indice is None:
                logging.warning(
                    "The measures indice is none. please set it using 'Kalmanfilter.populate_measures_constrain(self, indice)' later.")
            else:
                raise Exception(
                    "expect list or tuple, but encounter %s for measures_indice" % str(type(measures_indice)))
        return R

    def populate_states_variance_constrain(self, factors=None):
        """
        @return Q : np.array with shape of (states_dim, states_dim)
        """
        if factors is None:
            factors = np.zeros((self.states_dim, 1))
            factors[0, 0] = 1.0
        assert (factors.shape == (self.states_dim, 1))
        Q = factors * factors.T
        return Q

    ## Interfaces to modify the public attributes

    def set_dt(self, dt):
        self.dt = dt
        return self

    def set_P(self, P):
        self.P = P
        return self

    def set_Q(self, Q):
        self.Q = Q
        return self

    # @todo : TODO impl
    def predict(self, states):
        P = self.P
        A = self._A
        Q = self.Q

        # performance states prediction using pure physic models
        # I am going to embed opticalFlow control here, where magics happen
        # Suppose our moovement equation is not obeying `rigid body movement` but the estiated from
        # optcalFlow ? That's it!
        self._states = A.dot(states)

        self.P = A.dot(P.dot(A.T)) + Q
        return self._states

    # @todo : TODO impl
    # Not we can change measures dynamically for multi sensor fusion
    def update(self, observed_states):
        states = self._states
        P = self.P
        H = self.H
        R = self.R
        z = observed_states
        I = np.eye(self.states_dim)

        def validate_states(states):
            # check wheter states is an object implements python array alike protcol
            if not hasattr(states, "__len__"):
                raise TypeError("states should be an array like object!")
            assert (len(states) == self.measures_dim)

        validate_states(z)

        # perform states update
        # Kalman Gain from standard EKF theory
        K = (P.dot(H.T)).dot(np.linalg.pinv(H.dot(P.dot(H.T)) + R))
        # Update estimates
        self.states = states + K.dot(z - H.dot(states))
        # Update states error covariance
        self.P = (I - K.dot(H)).dot(P)
        return self.states


class BBoxKalmanFilter(ExtendedKalmanFilter):
    # specify states to observe
    # bounding box : (x, y, w, h) with shape equal to (4,1)
    observation_dim = 4
    states_dim = observation_dim * 3

    logger = LoggerAdaptor("BBoxKalmanFilter", _logger);

    def __init__(self):

        # we don't know velocity and accelertion of frame changing
        super().__init__(BBoxKalmanFilter.states_dim, BBoxKalmanFilter.observation_dim)

        self.bbox = np.zeros((4, 1))
        self._states = np.zeros((12, 1))

        self._Init()

    def _Init(self):
        # setup covariance matrix, borrow parameters by "Simple Object Realtime Tracking" directly]
        # since this is extremely hacky for different tasks
        self.setup_P()
        self.setup_Q()

        # initalize H and R
        self.H = self.populate_measures_constrain([0, 1, 2, 3])

        self.R = self.populate_priors_variance_constrain([0, 1, 2, 3])

    def setup_P(self, factors=None):
        if factors is None:
            transition_weight = 1. / 20
            velocity_weight = 1. / 160
            # freshly added
            acceleration_weight = 1. / 256
            factors_raw = np.array([transition_weight * 2, velocity_weight * 10, acceleration_weight * 10])
            factors = np.repeat(factors_raw, 4)  # x x x x y y y y z z z z

        self.P = np.diag(np.square(factors))

    def setup_Q(self, factors=None):
        if factors is None:
            transition_weight = 1. / 20
            velocity_weight = 1. / 160
            # freshly added
            acceleration_weight = 1. / 256
            factors_raw = np.array([transition_weight, velocity_weight, acceleration_weight])
            factors = np.repeat(factors_raw, 4)  # x x x x y y y y z z z z

        # also, you can use `populate_states_variance_constrain` method I provided
        # to populate the variance matrix
        self.Q = np.diag(np.square(factors))

    def predict(self, bbox):
        self.bbox = bbox
        self._states[:4, 0] = bbox
        ret = super().predict(self._states)
        ret = ret.reshape(ret.shape[0])
        return ret

    def update(self, observed_states):
        return super().update(np.array(observed_states).reshape((BBoxKalmanFilter.observation_dim, 1)))