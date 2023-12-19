import cv2
import easyocr
import numpy as np

def detect_objects(image):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            detected_objects.append(image[y:y + h, x:x + w])

    return detected_objects

def recognize_number_plate(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    
    if result:
        return result[0][1]
    else:
        return None

def display_results(frame, objects):
    for obj in objects:
        number_plate_text = recognize_number_plate(obj)
        if number_plate_text:
            print("Detected Number Plate:", number_plate_text)

        cv2.imshow('Number Plate Detection', obj)
        cv2.waitKey(0)

def main():
    source = input("Enter 'image' for image input or 'video' for video input: ")

    if source == 'image':
        image_path = input("Enter the path to the image file: ")
        frame = cv2.imread(image_path)
        objects = detect_objects(frame)
        display_results(frame, objects)

    elif source == 'video':
        video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam, or provide the path to a video file

        while True:
            ret, frame = video_capture.read()
            objects = detect_objects(frame)
            display_results(frame, objects)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
