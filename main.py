import os
import random

import cv2
import numpy as np
from ultralytics import YOLO

from tracker import Tracker

video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(0)

model = YOLO("yolov8n.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5

arr_in = set()
arr_out = set()
people_out_cnt = 0

# Get the screen width and height
screen_width = 1920  # Replace this with your screen width
screen_height = 1080  # Replace this with your screen height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (screen_width, screen_height))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
    #                           (frame.shape[1], frame.shape[0]))

    results = model(frame)

    # Calculate the dimensions of the rectangle
    rect_width = int(2 / 3 * screen_width)
    rect_height = int(0.5 * rect_width)  # You can adjust the height proportion if needed

    # Calculate the position to center the rectangle on the screen
    x = int((screen_width - rect_width) / 2)
    y = int((screen_height - rect_height) / 2)

    # Draw the rectangle on the image
    color = (0, 0, 255)  # BGR color (red)
    thickness = 2  # Line thickness
    cv2.rectangle(frame, (x, y), (x + rect_width, y + rect_height), color, thickness)

    x_left = x
    x_right = x + rect_width
    y_top = y
    y_bottom = y + rect_height

    cv2.putText(
        frame,
        "x_left: " + str(x_left) + " ,x_right: " + str(x_right) + " ,y_top: " + str(y_top) + " ,y_bottom: " + str(
            y_bottom),
        (x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            class_name = "etc."
            if class_id == 0:
                class_name = "people"

            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
                cv2.putText(
                    frame,
                    str(class_name),
                    (x2, y2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        tracker.update(frame, detections)

        count = 0

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            cv2.putText(
                frame,
                str(track_id),
                (int(x1), int(y1)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            if x1 >= x_left and x2 <= x_right and y1 >= y_top and y2 <= y_bottom:
                if track_id in arr_out:
                    people_out_cnt += 1
                    arr_out.remove(track_id)

                arr_in.add(track_id)
                count += 1

            else:
                if track_id in arr_in:
                    people_out_cnt -= 1
                    arr_in.remove(track_id)

                arr_out.add(track_id)

        cv2.putText(
            frame,
            "count: " + str(count),
            (10, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "people_out_cnt: " + str(people_out_cnt),
            (10, 300),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # cap_out.write(frame)
    cv2.imshow('Notebook Camera', frame)

cap.release()
# cap_out.release()
cv2.destroyAllWindows()
