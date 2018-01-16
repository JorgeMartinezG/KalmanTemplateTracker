import cv2
import cv2.cv as cv
import math
import optparse
import os


def update_params(obj, img, out_bbox, full_tpl):
    x_point, y_point = out_bbox[0], out_bbox[1]
    kalman, measurement = obj['kalman'], obj['measurement']

    measurement[0, 0] = x_point
    measurement[1, 0] = y_point

    kalman.state_pre[0, 0] = x_point 
    kalman.state_pre[1, 0] = y_point
    kalman.state_pre[2, 0] = 0
    kalman.state_pre[3, 0] = 0

    estimates = cv.KalmanCorrect(kalman, measurement)

    # Update template.
    img_tpl = get_template(obj, img, out_bbox, full_tpl)

    # Create kalman corrected bounding box.
    correct_bbox = create_correct_bbox(estimates, out_bbox)

    obj.update(dict(kalman=kalman,
                    measurement=measurement,
                    #estimates=estimates,
                    tpl=img_tpl))

    return obj, correct_bbox


def kalman_create():
    kalman = cv.CreateKalman(4, 2, 0)

    # set kalman transition matrix
    kalman.transition_matrix[0, 0] = 1
    kalman.transition_matrix[1, 1] = 1
    kalman.transition_matrix[2, 2] = 1
    kalman.transition_matrix[3, 3] = 1

    # set Kalman Filter
    cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
    cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(1e-6))
    cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(1e-2))
    cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(0.1))

    return kalman 


def detect_objects(classifier, img_frame):
    coords = classifier.detectMultiScale(
        img_frame,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return coords


def draw_rectangle(img_frame, coords, color):
    color_tuple = (0, 255, 0)
    if color == 'red':
        color_tuple = (0, 0, 255)

    x_point, y_point, width, height = coords
    cv2.rectangle(
        img_frame,
        (x_point, y_point),
        (x_point + width, y_point + height),
        color_tuple,
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


def create_correct_bbox(estimates, out_bbox):
    bbox = [estimates[0, 0], estimates[1,0]]
    bbox.extend(out_bbox[2:])
    bbox = [int(coord) for coord in bbox]

    return bbox


def get_obj_patch(img, obj_bbox):
    x_point, y_point, width, height = obj_bbox
    return img[y_point: y_point + height,
               x_point: x_point + width]


def create_object_dict(img, obj_bbox):
    x_point, y_point, _, _ = obj_bbox

    # Create Kalman filter.
    kalman = kalman_create()

    # Create measurement matrix.
    measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

    # Create object dictionary.
    obj = dict(kalman=kalman, measurement=measurement)

    # Update kalman params.
    obj = update_params(obj, img, obj_bbox, full_tpl=True)

    return obj


def create_pred_bbox(obj, prediction):
    # Get template width and height.
    height, width, _ = obj.get('tpl').shape

    return [prediction[0, 0], prediction[1, 0], width, height]


def l2_norm(point_1, point_2):
    return math.sqrt((point_2[1] - point_1[1])**2 + 
                     (point_2[0] - point_1[0])**2)


def get_closest_obj(pred_bbox, objs):
    # Compute distance wrt prediction bounding box.
    distances = [l2_norm(obj.tolist(), pred_bbox) for obj in objs]

    # If the smallest distance is higher than 50 pixels, return None and
    # template match.
    if min(distances) > 50:
        return None

    # Find obj with smallest distance.
    return objs[distances.index(min(distances))]


def get_capture_params(cap):
    return dict(width=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)),
                fourcc=cv2.cv.CV_FOURCC('P', 'I', 'M', '1'),
                fps=int(cap.get(cv2.cv.CV_CAP_PROP_FPS)))


def create_video_writer(cap, file_name, output_folder='output'):
    params = get_capture_params(cap)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # create video file.
    video_path = os.path.join(output_folder, file_name)
    writer = cv2.VideoWriter(video_path, params.get('fourcc'),
                             params.get('fps'),
                             (params.get('height'), params.get('width')))

    return writer


def main():
    # Read input options.
    options = parse_options()
    cap = cv2.VideoCapture(options.path)
    track_sw = False

    # Create detector.
    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder, 'Classifiers', 'haarcascade_frontalface_alt.xml')
    classifier = cv2.CascadeClassifier(path)

    if options.write:
        file_name = options.path.split('/')[-1]
        writer = create_video_writer(cap, file_name)

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
            pred_bbox = objs[0].tolist()
            obj, _ = create_object_dict(img, pred_bbox)
            track_sw = True
            continue

        # Create kalman prediction bounding box.
        prediction = cv.KalmanPredict(obj.get('kalman'))

        # Create prediction bounfing box based on template shape.
        pred_bbox = create_pred_bbox(obj, prediction)

        # In the detector we trust. 
        if len_objs != 0:
            full_tpl = True
            # Find detection whose distance to prediction is the smallest.
            out_bbox = get_closest_obj(pred_bbox, objs)

        # Detector did not find anything...run template match.
        if len_objs == 0 or out_bbox is None:
            full_tpl = False
            out_bbox = run_template_match(img, pred_bbox, obj)

        # Update Kalman and get bbox.
        obj, bbox = update_params(obj, img, out_bbox, full_tpl)

        draw_rectangle(img, bbox, 'green')
        if options.detections:
            [draw_rectangle(img, det, 'red') for det in objs]

        cv2.imshow('video', img)
        if options.write:
            writer.write(img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def parse_options():
    parser = optparse.OptionParser()
    parser.add_option('-p',
                      '--path',
                      dest='path')
    parser.add_option('-d',
                      '--det',
                      dest='detections',
                      action='store_true',
                      default=False)
    parser.add_option('-w',
                      '--write',
                      dest='write',
                      action='store_true',
                      default=False)

    options, _ = parser.parse_args()

    return options


def run_template_match(img, pred_bbox, obj):
    # Create search neighborhood around kalman prediction.
    search_patch, min_x, min_y = create_search_patch(img, pred_bbox)

    # Get previously saved template.
    template = obj.get('tpl')
    height, width, _ = template.shape

    # Run template match around search neighborhood.
    res = cv2.matchTemplate(search_patch, template, cv2.TM_CCOEFF)
    _, _, _, top_left = cv2.minMaxLoc(res)

    # Add the top left position from the search patch.
    top_left = top_left[0] + min_x, top_left[1] + min_y
    
    # Create object bounding box.
    out_bbox = list(top_left)
    out_bbox.extend([width, height])

    return out_bbox


def get_template(obj, img, bbox, full):
    img_tpl = get_obj_patch(img, bbox) 
    if not full:
        img_tpl = (obj['tpl'] * 0.9 + img_tpl * 0.1).astype('uint8')

    return img_tpl


if __name__ == '__main__':
    main()
