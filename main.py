import cv2
import cv2.cv as cv
import optparse
import os

#def create_cap():
#def get_or_create_helpers(face_coords, img_frame)


def update_params(obj, x_point, y_point):
    kalman, measurement = obj['kalman'], obj['measurement']

    measurement[0, 0] = x_point
    measurement[1, 0] = y_point

    kalman.state_pre[0, 0] = x_point 
    kalman.state_pre[1, 0] = y_point
    kalman.state_pre[2, 0] = 0
    kalman.state_pre[3, 0] = 0

    estimates = cv.KalmanCorrect(kalman, measurement)

    obj.update(dict(kalman=kalman,
                    measurement=measurement,
                    estimates=estimates))

    return obj 


def kalman_create():
    kalman = cv.CreateKalman(4, 2, 0)

    # set kalman transition matrix
    kalman.transition_matrix[0, 0] = 1
    kalman.transition_matrix[1, 1] = 1
    kalman.transition_matrix[2, 2] = 1
    kalman.transition_matrix[3, 3] = 1

    # set Kalman Filter
    cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
    cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(1e-5))
    cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(1e-1))
    cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(0.1))

    return kalman 


def detect_objects(classifier, img_frame):
    coords = classifier.detectMultiScale(
        img_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return coords


def draw_rectangle(img_frame, coords):
    x_point, y_point, width, height = coords
    cv2.rectangle(
        img_frame,
        (x_point, y_point),
        (x_point + width, y_point + height),
        (0, 255, 0),
        2
    )


def create_search_patch(img, obj_bbox, search_size=50):
    x_point, y_point, width, height = obj_bbox

    roi_min_y = max(0, y_point - search_size)
    roi_max_y = min(img.shape[1], y_point + height + search_size)
    roi_min_x = max(0, x_point - search_size)
    roi_max_x = min(img.shape[0], x_point + width + search_size)

    roi = img[roi_min_y:roi_max_y, roi_min_x:roi_max_x]

    return roi, roi_min_x, roi_min_y 


def get_obj_patch(img, obj_bbox):
    x_point, y_point, width, height = obj_bbox
    return img[y_point: y_point + height + 1,
               x_point: x_point + width + 1]


def create_object_dict(img, obj_bbox):
    x_point, y_point, _, _ = obj_bbox

    # Create Kalman filter.
    kalman = kalman_create()

    # Create measurement matrix.
    measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

    # Create image template patch.
    obj_patch = get_obj_patch(img, obj_bbox)

    # Create object dictionary.
    obj = dict(kalman=kalman, measurement=measurement, tpl=obj_patch)

    # Update kalman params.
    obj = update_params(obj, x_point, y_point)

    return obj


def main():
    # Read input options.
    #options = parse_options()
    cap = cv2.VideoCapture('gupta/c50.avi')
    track_sw = False

    # Create detector.
    classifier_path = 'classifiers/haarcascade_frontalface_alt.xml'
    classifier = cv2.CascadeClassifier(classifier_path)

    while cap.isOpened():
        # Retreive video frame from capture and status
        _, img = cap.read()
        objs = detect_objects(classifier, img)
        len_objs = len(objs)

        # Waiting for detections.
        if len_objs == 0 and not track_sw:
            cv2.imshow('Video', img); cv2.waitKey(25)
            continue

        # Initialize Kalman filter.
        elif len_objs != 0 and not track_sw:
            pred_bbox = objs[0]
            obj = create_object_dict(img, pred_bbox)
            track_sw = True
            continue

        prediction = cv.KalmanPredict(obj.get('kalman'))
        pred_bbox[:2] = prediction[0, 0], prediction[1, 0]
        bbox = pred_bbox.copy()

        if len_objs == 0:
            # Create neighborhood around kalman prediction. 
            search_patch, min_x, min_y = create_search_patch(img, pred_bbox)
            template = obj.get('tpl')
            height, width, _ = template.shape

            # Run template match.
            res = cv2.matchTemplate(search_patch, template, cv2.TM_CCOEFF)
            _, _, _, top_left = cv2.minMaxLoc(res)

            # Add the top left position from the search patch.
            top_left = top_left[0] + min_x, top_left[1] + min_y
            bottom_right = (top_left[0] + width, top_left[1] + height)
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            
            # Create new template.
            out_bbox = list(top_left)
            out_bbox.extend(pred_bbox[2:].tolist())
            obj = update_params(obj, out_bbox[0], out_bbox[1])

            # Partially Update template img.
            img_tpl = get_obj_patch(img, out_bbox) 
            new_tpl = obj['tpl'] * 0.7 + img_tpl * 0.3
            obj.update(dict(tpl=new_tpl.astype('uint8')))
            draw_rectangle(img, out_bbox)

        # We trust completely in the detector.
        if len_objs != 0:
            full_update = True
            # TODO: Check if bounding box distance to prediction is below th.
            out_bbox = objs[0]
            obj = update_params(obj, out_bbox[0], out_bbox[1])

            estimates = obj.get('estimates')
            bbox = [estimates[0, 0], estimates[1,0]]
            bbox.extend(pred_bbox[2:].tolist())
            bbox = [int(coord) for coord in bbox]

            # Fully Update template img.
            img_tpl = get_obj_patch(img, bbox) 
            obj.update(dict(tpl=img_tpl))

            draw_rectangle(img, bbox)

        cv2.imshow('video', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def parse_options():
    parser = optparse.OptionParser()
    parser.add_option('-p',
                      '--path',
                      dest='path')


if __name__ == '__main__':
    main()
