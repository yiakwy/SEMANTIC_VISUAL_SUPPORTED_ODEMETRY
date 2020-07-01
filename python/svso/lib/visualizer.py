import os
import sys
import argparse
import functools
import cv2
import numpy as np
CV_CUDA = False

print(cv2.__version__)

VERSION = cv2.__version__.split('.')
CV_MAJOR_VERSION = int(VERSION[0])

# print(cv2.CV_AA)

if not cv2.cuda.getCudaEnabledDeviceCount():
  try:
    mat_cpu = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
    mat_gpu = cv2.cuda_GpuMat()
    mat_gpu.upload(mat_cpu)
    CV_CUDA = True
    print("cuda is enabled for opencv")
  except Exception as e:
    print(e)
    print("You have to compile CUDA manually.")
    # @todo : TODO add cuda support

import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours

# logging.basicConfig(level=logging.INFO)

def add_path(path):
    path = os.path.abspath(path)
    if path not in sys.path:
        logging.info("load path %s" % path)
        sys.path.insert(0, path)

pwd = os.path.dirname(os.path.realpath(__file__))
logging.info(pwd)

# Add Paths Safely
add_path(os.path.join(pwd, '..', 'config'))
add_path(os.path.join(pwd, '.'))

# Add Modules

from svso.py_pose_graph.point3d import Point3D
#from config import Settings
from svso.config import Settings

class VisualizerConfig(Settings):
    pass

conf = VisualizerConfig("settings")
add_path(conf.MRCNN)

from mrcnn import visualize

from math import log
# visualization toolkits
def plot_vector_fields_from_pandas(df, title=None, dest=None, ax=None):
    data = df.as_matrix(columns=df.columns)
    x = data[:,0]
    y = data[:,1]
    dx= data[:,2]
    dy= data[:,3]
    figsize = (16, 16)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    if isinstance(title, str) and title is not "":
       ax.set_title(title)
    kw = {
    "units" : "inches",
    "scale_units": "xy",
    "angles": "xy",
    "pivot": "tip",
    "color": "g",
    "scale": 2,
    "width": 0.02
    }
    q = ax.quiver(x,y,dx,dy,**kw)
    ax.plot(x,y,"ro-")
    plt.show()
    if isinstance(dest, str) and dest is not "":
        fig.savefig(dest)
    pass

# @todo : TODO
def plot_vector_fields(x, y, u, v):
    pass

def display(im, ax=None):
    figsize = (16, 16)
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
    height, width = im.shape[:2]
    size=(width, height)
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.imshow(im.astype(np.uint8))


# simple image renderer suite
class WebImageRenderer:

    # implement a image renderer using opencv as backend
    def __init__(self):
        self.camera = None
        pass

    def drawMatchedROI(self, img, reference_img, mtched, unmtched_landmarks, unmtched_detections):

        # First : unmatched landmarks
        # Second : unmatched detections

        n_mtches = len(mtched) + 2
        colors = visualize.random_colors(n_mtches)

        if not n_mtches:
            logging.info("No instances to display!")
            return img

        def _apply_mask(image, mask, color, alpha=0.5):
            """Apply the given mask to the image.
            """
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c] * (1 - alpha) + alpha * color[c],
                                          image[:, :, c])
            return image

        def _drawROI(image, box, mask, color, label, score, _id):

            masked_image = image.copy()

            # Bounding box
            if not np.any(box):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                return masked_image

            y1, x1, y2, x2 = box

            caption = "<Landmark #{} : {}({:.3f})>".format(_id, label, score) if score \
                else "<landmark #{} : {}>".format(_id, label)

            masked_image = visualize.apply_mask(masked_image, mask, color)
            masked_image_with_boxes = cv2.rectangle(masked_image, (x1, y1), (x2, y2), np.array(color) * 255, 2)

            # Mask Polygon
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8
            )
            padded_mask[1:-1, 1:-1] = mask
            # contours = find_contours(padded_mask, 0.5)
            if CV_MAJOR_VERSION > 3:
                contours, _ = cv2.findContours(padded_mask,
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                _, contours, _ = cv2.findContours(padded_mask,
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            masked_image_with_contours_plus_boxes = cv2.drawContours(masked_image_with_boxes, contours, -1, (0, 255, 0),
                                                                     1)

            out = cv2.putText(
                masked_image_with_contours_plus_boxes, caption, (x1, y1 - 4), cv2.FONT_HERSHEY_PLAIN, 0.8,
                np.array(color) * 255, 1
            )

            masked_image = out
            return out

        MATCHED_COLORS = colors[1:-1]
        # print("mtched colors", MATCHED_COLORS)
        UNMATCHED__LANDMARK_COLORS = colors[0]
        UNMATCHED_DETECTION_COLORS = colors[-1]

        masked_img = img.copy()
        masked_reference_img = reference_img.copy()

        # drawing extraction results
        offset = 0
        for mtch in mtched:
            landmark, detection = mtch
            landmark.color = landmark.color or MATCHED_COLORS[offset]
            masked_reference_img = _drawROI(masked_reference_img,
                                            landmark.roi_features['box'],
                                            landmark.roi_features['mask'],
                                            landmark.color,
                                            landmark.label,
                                            landmark.score,
                                            landmark.seq)
            masked_img = _drawROI(masked_img,
                                  detection.roi_features['box'],
                                  detection.roi_features['mask'],
                                  landmark.color,
                                  detection.label,
                                  detection.score,
                                  landmark.seq)
            offset += 1

        for unmtched_landmark in unmtched_landmarks:
            masked_reference_img = _drawROI(masked_reference_img,
                                            unmtched_landmark.roi_features['box'],
                                            unmtched_landmark.roi_features['mask'],
                                            (0., 1., 1.),  # Yellow
                                            unmtched_landmark.label,
                                            unmtched_landmark.score,
                                            unmtched_landmark.seq)

        for unmtched_detection in unmtched_detections:
            masked_img = _drawROI(masked_img,
                                  unmtched_detection.roi_features['box'],
                                  unmtched_detection.roi_features['mask'],
                                  (0., 0., 1.),  # Red
                                  unmtched_detection.label,
                                  unmtched_detection.score,
                                  unmtched_detection.seq)

        # store the rendered images
        self._masked_img = masked_img
        self._masked_reference_img = masked_reference_img

        # drawing bbox matching results
        r1, c1 = masked_img.shape[0], masked_img.shape[1]
        r2, c2 = masked_reference_img.shape[0], masked_reference_img.shape[1]

        out = np.zeros((max([r1, r2]), c1 + c2, 3), dtype='uint8')

        out[:r1, :c1] = np.dstack([masked_img])
        out[:r2, c1:] = np.dstack([masked_reference_img])

        # draw line between matched bbox
        offset = 0
        for mtch in mtched:
            color = mtch[0].color or np.array(MATCHED_COLORS[offset]) * 255
            y1_1, x1_1, y2_1, x2_1 = mtch[1].roi_features['box']
            cy_1 = (y2_1 + y1_1) / 2.0
            cx_1 = (x2_1 + x1_1) / 2.0
            y1_2, x1_2, y2_2, x2_2 = mtch[0].roi_features['box']
            cy_2 = (y2_2 + y1_2) / 2.0
            cx_2 = (x2_2 + x1_2) / 2.0

            # draw lines
            cv2.line(out, (x1_1, y1_1), (x1_2 + c1, y1_2), color, 1)
            cv2.line(out, (int(x2_1), int(y1_1)), (int(x2_2) + c1, int(y1_2)), color, 1)
            cv2.line(out, (int(x2_1), int(y2_1)), (int(x2_2) + c1, int(y2_2)), color, 1)
            cv2.line(out, (int(x1_1), int(y2_1)), (int(x1_2) + c1, int(y2_2)), color, 1)

            # cv2.line(out, (int(cx_1),int(cy_1)), (int(cx_2)+c1,int(cy_2)), color, 1)

            offset += 1

        return out

    def drawMatchesKnn(self, img1, kps1, img2, kps2, kps_mtched1to2, mask):
        draw_params = dict(matchColor=(0, 255, 0),  # G
                           singlePointColor=(255, 0, 0),  # R
                           # this is important to filter out the dense connection
                           matchesMask=mask,
                           flags=0)

        masked_img = cv2.drawMatchesKnn(img1, kps1, img2, kps2, kps_mtched1to2, None, **draw_params)
        return masked_img

    # Credits to the original author:
    #   https://www.hongweipeng.com/index.php/archives/709/
    #   https://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
    def drawMatches(self, img1, kp1, img2, kp2, matches):
        """
        My own implementation of cv2.drawMatches as OpenCV 2.4.9
        does not have this function available but it's supported in
        OpenCV 3.0.0

        This function takes in two images with their associated
        keypoints, as well as a list of DMatch data structure (matches)
        that contains which keypoints matched in which images.

        An image will be produced where a montage is shown with
        the first image followed by the second image beside it.

        Keypoints are delineated with circles, while lines are connected
        between matching keypoints.

        img1,img2 - Grayscale images
        kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
                  detection algorithms
        matches - A list of matches of corresponding keypoints through any
                  OpenCV keypoint matching algorithm
        """

        # Create a new output image that concatenates the two images together
        # (a.k.a) a montage
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]

        out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

        # Place the first image to the left
        out[:rows1, :cols1] = np.dstack([img1])

        # Place the next image to the right of it
        out[:rows2, cols1:] = np.dstack([img2])

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for mat in matches:
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
            cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

        return out

    def drawPredictions(self, ref, kps1, cur, kps2, matches):
        frame_key = ref.seq
        img_shp = cur.img.shape[0:2]

        #     optical_flow = OpticalFlowKPntPredictor()
        #     optical_flow.Init()
        #     optical_flow.set_FromImg(ref.img_grey())

        # compute accumulative flows
        pre = cur.pre
        flow = pre.predictors['OpticalFlowKPnt'].get_flow()
        assert flow.shape[:2] == img_shp
        while pre.seq is not frame_key:
            pre = pre.pre
            flow1 = pre.predictors['OpticalFlowKPnt'].get_flow()
            assert flow.shape == flow1.shape

            # flow += flow1

            flow2 = np.zeros((img_shp[0], img_shp[1], 2))
            for y in range(img_shp[0]):
                for x in range(img_shp[1]):
                    y1 = y + flow[int(y), int(x)][1]
                    x1 = x + flow[int(y), int(x)][0]
                    if int(y1) >= img_shp[0] or int(y1) < 0 or \
                            int(x1) >= img_shp[1] or int(x1) < 0:
                        continue
                    flow2[y, x] = flow1[int(y1), int(x1)]

            flow = flow2

        out = cur.img.copy()
        for mat in matches:
            # Get the matching keypoints for each of the images
            img1_idx = mat.trainIdx
            img2_idx = mat.queryIdx

            # x - columns
            # y - rows
            (x1, y1) = kps1[img1_idx].pt
            (x2, y2) = kps2[img2_idx].pt

            pred = (x1 + flow[int(y1), int(x1), 0], y1 + flow[int(y1), int(x1), 1])
            #         pred = optical_flow.predict([[x1,y1]], cur.img_grey())[0]

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            # cv2.circle(out, (int(x1),int(y1)), 4, (0, 0, 255), 1)
            cv2.circle(out, (int(pred[0]), int(pred[1])), 4, (0, 0, 255), 1)
            cv2.circle(out, (int(x2), int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(pred[0]), int(pred[1])), (int(x2), int(y2)), (0, 0, 255), 1)

        return out

    def drawOpticalFlow(self, frame, step=16):
        flow = frame.predictors["OpticalFlowKPnt"].get_flow()
        H, W = frame.img.shape[:2]
        y, x = np.mgrid[step / 2:H:step, step / 2:W:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        mask = np.zeros_like(frame.img)
        cv2.polylines(mask, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(mask, (x1, y1), 1, (0, 255, 0), -1)
        return mask

    def drawAxis(self, frame=None):
        # unit is mm
        arrow_length = 0.3
        dv = np.array([0., 0., 0.])

        # compute the point to place an axis object, R, t are estiamted from ransac and posegraph optimization

        H, W = frame.img.shape[:2]
        rot_vec, _ = cv2.Rodrigues(frame.R0)
        points = np.float32([[arrow_length, 0 + dv[1], 0 + dv[2]],
                             [0, arrow_length + dv[1], 0 + dv[2]],
                             [0, 0 + dv[1], arrow_length + dv[2]],
                             [0, 0 + dv[1], 0 + dv[2]]]).reshape(-1, 3)
        # set distortion to zeros
        axisPoints, _ = cv2.projectPoints(points, rot_vec, frame.t0, frame.camera.K, (0, 0, 0, 0))
        print("Axis in image pixels (OpenCV): \n%s" % axisPoints)
        imagepoints = []
        for p in points:
            cam_pt = frame.camera.viewWorldPoint(Point3D(p[0], p[1], p[2]))
            px = frame.camera.view(cam_pt)
            imagepoints.append(px)
            pass
        imagepoints = np.array(imagepoints)
        print("images points computed: \n%s" % imagepoints)
        mask = np.zeros_like(frame.img)
        #     mask = cv2.line(mask, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 5)
        #     mask = cv2.line(mask, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 5)
        #     mask = cv2.line(mask, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 5)

        mask = cv2.line(mask, tuple(imagepoints[3].data), tuple(imagepoints[0].data), (255, 0, 0), 5)
        mask = cv2.line(mask, tuple(imagepoints[3].data), tuple(imagepoints[1].data), (0, 255, 0), 5)
        mask = cv2.line(mask, tuple(imagepoints[3].data), tuple(imagepoints[2].data), (0, 0, 255), 5)
        return mask

    def drawGrid(self, frame=None, gridNum=8, gridSize=0.1):
        camera = frame.camera or self.camera

        if camera is None:
            raise Exception("Camera should not be None!")

        H, W = frame.img.shape[:2]

        halfGridNum = (gridNum - 1) / 2
        grid = np.zeros((gridNum, gridNum, 2))
        dv = np.array([0., 0., 0.])

        for i in range(gridNum):
            for j in range(gridNum):
                x = (i - halfGridNum) * gridSize + dv[0]
                y = (j - halfGridNum) * gridSize + dv[1]
                z = 0. + dv[2]

                v = np.array([x, y, z]).reshape(3, 1) - frame.t0
                pt_cam = np.linalg.inv(frame.R0).dot(v)
                pt_cam = pt_cam.reshape((3,))
                if np.abs(pt_cam[2]) < 0.001:
                    pt_cam[2] = 0.001
                px = camera.view(Point3D(pt_cam[0], pt_cam[1], pt_cam[2]))
                if px is None:
                    grid[i, j, 0] = -1
                    grid[i, j, 1] = -1
                    continue
                grid[i, j, 0] = px.x
                grid[i, j, 1] = px.y

        # draw grid
        mask = np.zeros_like(frame.img)
        for i in range(gridNum):
            for j in range(gridNum - 1):
                if grid[i, j, 0] == -1 and grid[i, j, 1] == -1:
                    continue
                if grid[i, j + 1, 0] == -1 and grid[i, j + 1, 1] == -1:
                    continue
                mask = cv2.line(mask, (int(grid[i, j, 0]), int(grid[i, j, 1])),
                                (int(grid[i, j + 1, 0]), int(grid[i, j + 1, 1])), (255, 0, 255), 1)

        for j in range(gridNum):
            for i in range(gridNum - 1):
                if grid[i, j, 0] == -1 and grid[i, j, 1] == -1:
                    continue
                if grid[i + 1, j, 0] == -1 and grid[i + 1, j, 1] == -1:
                    continue
                mask = cv2.line(mask, (int(grid[i, j, 0]), int(grid[i, j, 1])),
                                (int(grid[i + 1, j, 0]), int(grid[i + 1, j, 1])), (255, 0, 255), 1)

        return mask

    def render(self, im, mode=None):
        if mode is 'webcam':
            try:
                from google.colab.patches import cv2_imshow
            except Exception as e:
                logging.warning(e)

                def wrapped_cv_img_render(img):
                    cv2.imshow(mode, img)

                cv2_imshow = wrapped_cv_img_render

            cv2_imshow(im)
            # wait for highgui processing drawing requests from cv::show
            k = cv2.waitKey(30) & 0xff
            if k == 'ESC':
                raise StopIteration()
        else:
            logging.warn("Use matplotlib as image rendering backend")
            figsize = (16, 16)
            _, ax = plt.subplots(1, figsize=figsize)
            height, width = im.shape[:2]
            size = (width, height)
            ax.set_ylim(height + 10, -10)
            ax.set_xlim(-10, width + 10)
            ax.axis('off')
            ax.imshow(im.astype(np.uint8))
            # show and clear the image cache immediately
            plt.show()


def Program(raw_args):
    task_id = "204272297"
    case_no = "15"
    file_name = os.path.join(conf.DATA_DIR, task_id, case_no, "vector_fields.csv")
    dest = os.path.join(conf.DATA_DIR, task_id, case_no, "Vector Fields(First Order Precision).png")
    shape = pd.read_csv(file_name)
    print("Total %s lines, snippet of top 1o (or less) records" % shape.shape[0])
    print(shape.head(10))
    plot_vector_fields_from_pandas(shape, title="Vector Fields(First Order Precision)", dest=dest)

if __name__ == "__main__":
    # sys.exit(Program(sys.argv[1:]))
    pass
