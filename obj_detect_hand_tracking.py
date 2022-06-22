import cv2
import mediapipe as mp
import numpy as np
from numba import njit


def load_yolo(weights_path, cfg_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [
        layers_names[i - 1] for i in net.getUnconnectedOutLayers()
    ]
    return net, classes, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=0.00392, size=(224, 224),
        mean=(0, 0, 0), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


@njit(cache=True)
def get_box_dimensions(outputs, width, height, box_conf):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            score = scores[class_id]
            if score > box_conf and class_id in [0, 16]:  # 人と犬
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(score))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img, indexes):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 1)


def hand_tracking_from_camera():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)

    model, classes, output_layers = load_yolo(
        weights_path='models/yolov4.weights',
        cfg_path='models/yolov4.cfg',
        names_path='models/coco.names'
    )

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image.flags.writeable = True

            height, width, channels = image.shape
            blob, outputs = detect_objects(image, model, output_layers)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            boxes, confs, class_ids = get_box_dimensions(
                outputs, width, height, box_conf=0.3
            )
            indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
            draw_labels(
                boxes, confs, colors, class_ids, classes, image, indexes
            )

            cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    hand_tracking_from_camera()
