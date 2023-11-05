import cv2
import numpy as np
#import facenet_pytorch
import numpy
#from random import randint

from detect import detect_face, predict_keypoints
#from shuffle_keypoint import predict_keypoints_onnx




class FaceSwaper:
    def __init__(self, max_size_img = 1000):
        self.MAX_SIZE_IMAGE = max_size_img
        pass


    def transformation_from_points(self, points1, points2):
        points1 = points1.astype(numpy.float64)
        points2 = points2.astype(numpy.float64)

        c1 = numpy.mean(points1, axis=0)
        c2 = numpy.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        c1 = c1.reshape(1, 2)
        c2 = c2.reshape(1, 2)

        s1 = numpy.std(points1)
        s2 = numpy.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = numpy.linalg.svd(points1.T @ points2)
        R = (U @ Vt).T

        return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                           c2.T - (s2 / s1) * R @ c1.T)),
                             numpy.matrix([0., 0., 1.])])

    def warp_im(self, im, M, dshape):
        output_im = numpy.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im

    def get_landmark_points(self, _img):
        max_size = np.max(_img.shape)
        img = np.zeros((max_size, max_size, 3))
        img[:_img.shape[0], :_img.shape[1]] = _img
        x1, y1, x2, y2 = detect_face(img)[0].astype(int)
        #delta_h, delta_w = abs(x2 - x1) // 4, abs(y2 - y1) // 4
        h = min(x2-x1, y2-y1)
        img = img[y2-h:y2, x2-h:x2]
        key_point = predict_keypoints(img).reshape((-1, 2))
        #self.DrawKeyPoints(cv2.resize(img.astype(np.uint8), (max(img.shape), max(img.shape))), key_point.astype(int))
        key_point[:, 0] = key_point[:, 0]*img.shape[1]/np.max(img.shape) + x2-h
        key_point[:, 1] = key_point[:, 1]*img.shape[0]/np.max(img.shape) + y2-h

        return key_point

    def align_perspective(self, img1, img2, landmarks1, landmarks2, eps = 0.1, k = 1.5):
        # определяем ракурс первого лица
        delta_eyes_right = abs(landmarks1[97, 0] - landmarks1[54, 0]) / k
        delta_eyes_left = abs(landmarks1[96, 0] - landmarks1[54, 0]) / k
        delta_eyes = max(delta_eyes_right, delta_eyes_left)
        delta1 = (delta_eyes_right - delta_eyes_left) / delta_eyes

        # определяем ракурс второго  лица
        delta_eyes_right = abs(landmarks2[97, 0] - landmarks2[54, 0]) / k
        delta_eyes_left = abs(landmarks2[96, 0] - landmarks2[54, 0]) / k
        delta_eyes = max(delta_eyes_right, delta_eyes_left)
        delta2 = (delta_eyes_right - delta_eyes_left) / delta_eyes

        if delta1 * delta2 < 0 and (abs(delta1) + abs(delta1)) / 2 > eps:
            return cv2.flip(img2, 1)
        else:
            return img2

    def align_face(self, im1, im2, landmarks1, landmarks2):
        def calculate_ratio_landmark(landmarks):
            x1 = np.min(landmarks[:, 0])
            x2 = np.max(landmarks[:, 0])
            y1 = np.min(landmarks[:, 1])
            y2 = np.max(landmarks[:, 1])

            w = x2-x1
            h = y2-y1
            k = w/h
            return k

        k1 = calculate_ratio_landmark(landmarks1)
        k2 = calculate_ratio_landmark(landmarks2)

        k=k1/k2
        im2 = cv2.resize(im2, (int(im2.shape[1]/k), im2.shape[0]))
        im2=self.normalize_image_shape(im2, im2.shape)
        return im2

    def correct_landmark(self, landmarks, factor = 0.9):
        landmarks[34: 38, 1] = landmarks[34: 38, 1]*factor
        landmarks[42: 46, 1]=landmarks[42: 46, 1]*factor
        return landmarks





    def draw_keypoints(self, img, points):
        import matplotlib.pyplot as plt
        plt.imshow(img)
        for x, y in points:
            plt.scatter(x, y, color='red')
        plt.show()

    def normalize_image_shape(self, img, shape):
        h, w, channels = img.shape
        max_size = max(shape)
        #k_w = randint(1, 10)
        right = (max_size - w) // 2
        right_matrix = np.zeros((h, right, channels)).astype(np.uint8)
        left = max_size - w - right
        left_matrix = np.zeros((h, left, channels)).astype(np.uint8)
        img = np.hstack((left_matrix, img, right_matrix)).astype(np.uint8)

        up = (max_size - h) // 2
        up_matrix = np.zeros((up, max_size, channels)).astype(np.uint8)
        down = max_size - h - up
        down_matrix = np.zeros((down, max_size, channels)).astype(np.uint8)
        img = np.vstack((up_matrix, img, down_matrix)).astype(np.uint8)

        return img

    def correct_colours(self, im1, im2, landmarks1):
        COLOUR_CORRECT_BLUR_FRAC = 1.0

        blur_amount = abs(landmarks1[96][0] - landmarks1[97][0]) / 2 #.228

        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0).astype(np.int32)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0).astype(np.int32)

        # Avoid divide-by-zero errors.
        im2_blur += 128 * (im2_blur <= 1.0)

        return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                (im2_blur.astype(numpy.float64)).astype(np.uint8))

    def draw_convex_hull(self, im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)


    def get_face_mask(self, im, landmarks):

        FEATHER_AMOUNT = int(abs(landmarks[96, 0] - landmarks[97, 0]) // 5)  # //5
        # print(FEATHER_AMOUNT)
        if FEATHER_AMOUNT % 2 == 0: FEATHER_AMOUNT += 1

        im = np.zeros(im.shape[:2], dtype=numpy.float64)

        LEFT_BROW_POINTS = list(range(33, 37))
        RIGHT_BROW_POINTS = list(range(42, 47))
        MOUTH_POINTS = list(range(76, 95))
        FACE_POINTS = list(range(1, 32))
        OVERLAY_POINTS = [
            LEFT_BROW_POINTS + RIGHT_BROW_POINTS + RIGHT_BROW_POINTS + MOUTH_POINTS
        ]

        self.draw_convex_hull(im,landmarks[OVERLAY_POINTS].astype(int),color=1)
        im = numpy.array([im, im, im]).transpose((1, 2, 0))

        for i in range(3):
            im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

        return im

    def ColorScale(self, img):
        for i in range(3):
            img[:, :, i] = np.where(img[:, :, i] > 255, 255, img[:, :, i])

        return img

    def image_form_corrected(self, img, landmarks1, landmarks2):
        def get_delta(landmark):
            x1, y1 = landmark[0]
            x2, y2 = landmark[3]

            return [np.sqrt((x1 - x2)**2 + (y1 - y2)**2), x1 - x2, y1 - y2]

        delta_1, _, __ = get_delta(landmarks1)
        delta_2, dx, dy = get_delta(landmarks2)


        sin = dy/delta_2
        cos = dx/delta_2

        rx = delta_2 * cos
        ry =  delta_2 * sin

        if delta_1 > delta_2:
            img = cv2.resize(img, (int(img.shape[1] + rx), int(img.shape[0] + ry)))
        else:
            img = cv2.resize(img, (int(img.shape[1] - rx), int(img.shape[0] - ry)))



        return img


    def swap(self, im1, im2):
        if len(im1.shape) == 2:
            im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
        if len(im2.shape) == 2:
            im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)

        if max(im1.shape) > self.MAX_SIZE_IMAGE: im1 = cv2.resize(im1, (int(im1.shape[1]*(max(im1.shape)/self.MAX_SIZE_IMAGE)),int(im1.shape[0]*(max(im1.shape)/self.MAX_SIZE_IMAGE))))
        if max(im2.shape) > self.MAX_SIZE_IMAGE: im2 = cv2.resize(im2, (int(im2.shape[1] // (max(im2.shape) / self.MAX_SIZE_IMAGE)), int(im2.shape[0] // (max(im2.shape) / self.MAX_SIZE_IMAGE))))
        history_shape = im1.shape

        im1 = self.normalize_image_shape(im1, im1.shape)
        im2 = self.normalize_image_shape(im2, im2.shape)

        if max(im1.shape) > max(im2.shape):
            im2 = self.normalize_image_shape(im2, im1.shape)  # make shape im1 equal to shape im2
        #try:
        landmarks1 = np.array(self.get_landmark_points(im1))  # Find facial landmarks points
        #except:
        #    return [True, 'first']
        try:
            landmarks2 = np.array(self.get_landmark_points(im2))
        except:
            return [True, 'second']


        im2 = self.align_perspective(im1, im2, landmarks1, landmarks2)
        #im2 = self.align_face(im1, im2, landmarks1, landmarks2)
        landmarks2 = np.array(self.get_landmark_points(im2))

        #M = self.transformation_from_points(landmarks1[[0, 1, 3, 4]],
        #                                    landmarks2[[0, 1, 3, 4]])  # Find transformation matrix
        M = self.transformation_from_points(landmarks1, landmarks2)  # Find transformation matrix
        # print(landmarks1.shape, landmarks2.shape)
        warped_im2 = self.warp_im(im2, M, im2.shape)[:im1.shape[0],
                     :im1.shape[1]]  # get transformation second-face image

        landmarks1 = self.correct_landmark(landmarks1)
        landmarks2 = self.correct_landmark(landmarks2)

        # Находим более норм фото
        if False:
            im2 = self.image_form_corrected(im2, landmarks1, landmarks2)
            im2 = self.normalize_image_shape(im2, im2.shape)

            if max(im1.shape) > max(im2.shape):
                im2 = self.normalize_image_shape(im2, im1.shape)  # make shape im1 equal to shape im2

            landmarks2 = np.array(self.get_landmark_points(im2))

            M = self.transformation_from_points(landmarks1[[0, 1, 3, 4]],
                                                landmarks2[[0, 1, 3, 4]])  # Find transformation matrix
            # print(landmarks1.shape, landmarks2.shape)
            warped_im2 = self.warp_im(im2, M, im2.shape)[:im1.shape[0],
                         :im1.shape[1]]  # get transformation second-face image

        #####


        # show_image(warped_im2)
        warped_corrected_im2 = self.correct_colours(im1, warped_im2, landmarks1)  # get swap second-face image
        # with second-face color
        warped_corrected_im2 = self.ColorScale(warped_corrected_im2)

        # show_image(warped_corrected_im2.astype(np.uint8))

        mask = self.get_face_mask(im2, landmarks2)  # Get mask
        #show_image(mask*255)
        warped_mask = self.warp_im(mask, M, im1.shape)  # get transformation second-face mask
        combined_mask = numpy.min([self.get_face_mask(im1, landmarks1), warped_mask],  # union masks
                                  axis=0)
        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask  # get swap image
        #self.ShowImage(output_im.astype(np.uint8))

        min_size = min(history_shape[:2])
        max_size = max(history_shape[:2])
        arg_min = np.argmin(history_shape[:2])
        if arg_min == 0:
            output_im = output_im[(max_size - min_size)//2: max_size - (max_size - min_size)//2, :]
        if arg_min == 1:
            output_im = output_im[:, (max_size - min_size)//2: max_size - (max_size - min_size)//2]

        return output_im.astype(np.uint8)

def show_image(img):
    import matplotlib.pyplot as plt
    plt.imshow(img.astype(np.uint8))
    plt.show()


if __name__ == '__main__' :
    root = 'D:/datasets/wiki/02'
    #img1 = cv2.imread(root + '/217702_1956-01-21_2013.jpg')
    img1 = cv2.imread(root + '/46402_1951-02-15_2009.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.imread(root + '/217702_1956-01-21_2013.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    show_image(img1)
    show_image(img2)

    swaper = FaceSwaper()
    img_result = swaper.swap(img1, img2)

    import matplotlib.pyplot as plt
    plt.imshow(img_result)
    plt.show()