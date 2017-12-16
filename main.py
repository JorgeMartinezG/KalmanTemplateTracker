import cv2
import optparse
import os

#def create_video_capture():
#def get_or_create_helpers(face_coords, img_frame)



def detect_faces(classifier, img_frame):
    faces_coords = classifier.detectMultiScale(
        img_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces_coords


def draw_rectangle(img_frame, coords):
    x_point, y_point, width, height = coords
    cv2.rectangle(
        img_frame,
        (x_point, y_point),
        (x_point + width, y_point + height),
        (0, 255, 0),
        2
    )


def main():
    # Read input options.
    #options = parse_options()
    video_capture = cv2.VideoCapture('gupta/c43.avi')

    # Create Haar-cascade classifier
    classifier_path = 'Classifiers/haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(classifier_path)

    while 1:
        # Retreive video frame from capture and status
        status, frame = video_capture.read()

        # Stop loop when the video has finished.
        if not status:
            break

        # Find faces using Haar cascade classifier.
        faces = detect_faces(face_cascade, frame)

        # Create region of interest (roi) and face patch given a coordinate.
        for face in faces:
            x_point, y_point, width, height = face

            # Create roi nearby detected face.
            test_val = 50
            roi = frame[y_point - test_val: y_point + height + test_val,
                        x_point - test_val: x_point + width + test_val]
            test_val = 0
            face_patch = frame[y_point - test_val: y_point + height + test_val,
                               x_point - test_val: x_point + width + test_val]
            cv2.imshow('roi', roi)
            cv2.imshow('face_patch', face_patch)

        #roi, patch = get_or_create_helpers(faces, frame)
        
        # Draw a rectangle around the faces
        for face_coords in faces:
            draw_rectangle(frame, face_coords)

        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def parse_options():
    parser = optparse.OptionParser()
    parser.add_option('-p',
                      '--path',
                      dest='path')


if __name__ == '__main__':
    main()