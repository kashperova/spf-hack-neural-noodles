from copy import deepcopy
import torch
from flask import Flask, render_template, request, redirect, send_file, url_for
import shutil
import os
import cv2
import glob
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.data.augment import LetterBox
import zipfile

# intialize Flask application
app = Flask(__name__, static_url_path='/static',
            static_folder='static',
            template_folder='templates')

# load trained model
model = YOLO('model.pt')

# Constants:
# 1. confidence threshold
# 2. counter for max units
# 3. filename of uploaded video
# 4. fps of uploaded video
CONF_THRESHOLD = 0.42
MAX_COUNT = 0
FILE = ""
FPS = 5


def plot_bb(new_boxes, new_masks, new_probs, orig_img):
    """

    :param new_boxes: bounding boxes with conf > CONF_THRESHOLD
    :param new_masks: masks of images with conf > CONF_THRESHOLD
    :param new_probs: probs of bounding boxes with conf > CONF_THRESHOLD
    :param orig_img: image for plotting bounding boxes
    :return: annotator.result :
    """
    line_width = 5
    font_size = 10
    font = 'Arial.ttf'
    pil = False
    img_gpu = None
    names = {0: 'soldier'}

    annotator = Annotator(deepcopy(orig_img), line_width, font_size, font, pil, example=names)
    pred_boxes, show_boxes = new_boxes, True
    pred_masks, show_masks = new_masks, True
    pred_probs, show_probs = new_probs, True

    # if we want to show masks of image
    if pred_masks and show_masks:
        if img_gpu is None:
            img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
            img_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
                2, 0, 1).flip(0).contiguous() / 255
        idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
        annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=img_gpu)

    # if we want to show bounding boxes on image
    if pred_boxes and show_boxes:
        for d in pred_boxes:
            p1, p2 = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
            cv2.rectangle(annotator.im, p1, p2, (0, 255, 0), thickness=5, lineType=cv2.LINE_AA)

    # if we want to show probabilities of
    # bounding boxes prediction on image
    if pred_probs is not None and show_probs:
        n5 = min(len(names), 5)
        top5i = pred_probs.argsort(0, descending=True)[:n5].tolist()  # top 5 indices
        text = f"{', '.join(f'{names[j] if names else j} {pred_probs[j]:.2f}' for j in top5i)}, "
        annotator.text((32, 32), text, txt_color=(255, 255, 255))
    return annotator.result()


def detect_units(video_path, video_check):
    """
    Method for detecting units on video frames and calculating
    the maximum number of units per video

    :param video_check: bool [download labeled video or not]
    :param video_path: str
    :return: None
    """

    global CONF_THRESHOLD
    global FPS
    global FILE

    # Initialize the counters
    global MAX_COUNT
    count = 0
    idx = 0

    if video_check is False:
        # Open the video file
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        FPS = fps

        # Set the number of frames to skip between each captured frame
        frames_to_skip = int(fps/2)

        # Set the starting frame to 0
        frame_num = 0
        # Loop over the frames in the video
        while True:
            # Read the next frame
            ret, frame = video.read()
            count = 0
            if not ret:
                break

            if frame_num % frames_to_skip == 0:

                # Run the object detection algorithm on the frame
                res = model.predict(source=frame, save=False)

                # remain only those boxes, the confidence threshold is greater
                # than the set value in the constant
                filtered_boxes = []
                for i in range(len(res[0].boxes.data)):
                    conf = res[0].boxes.conf[i]
                    if conf > CONF_THRESHOLD:
                        filtered_boxes.append([int(j) for j in res[0].boxes.data[i]])
                        count += 1
                        if count > MAX_COUNT:
                            MAX_COUNT = count
                # save only images with at least one bounding box
                if len(filtered_boxes) != 0:
                    res_plotted = plot_bb(filtered_boxes, res[0].masks, res[0].probs, res[0].orig_img)
                    cv2.imwrite("static/images/frame" + str(idx) + ".png", res_plotted)
                    idx += 1
            frame_num += 1
    else:
        results = model.predict(source=video_path, save=True, imgsz=1280, conf=CONF_THRESHOLD)
        os.rename("runs/detect/predict/" + FILE, "uploads/" + FILE)
        shutil.rmtree('runs')

        for i in range(len(results)):
            filtered_boxes = []
            count = 0
            for j in range(len(results[i].boxes.data)):
                conf = results[i].boxes.conf[j]
                if conf > CONF_THRESHOLD:
                    filtered_boxes.append([int(j) for j in results[i].boxes.data[j]])
                    count += 1
                    if count > MAX_COUNT:
                        MAX_COUNT = count
            if len(filtered_boxes) != 0:
                res_plotted = plot_bb(filtered_boxes, results[i].masks, results[i].probs, results[i].orig_img)
                cv2.imwrite("static/images/frame" + str(idx) + ".png", res_plotted)
                idx += 1


@app.route('/results', methods=['POST', 'GET'])
def results(video_check=False):
    if request.method == 'POST':
        if request.form["submit_button"] == "Завантажити серію зображень":
            # create zip archive for downloading
            shutil.make_archive('uploads/detected_images', 'zip', 'static/images/')
            return send_file('uploads/detected_images.zip', as_attachment=True)

        elif request.form["submit_button"] == "Завантажити розмічене відео":
            global FILE
            return send_file('uploads/' + FILE, as_attachment=True)
    if video_check:
        return render_template("results.html", max_count=MAX_COUNT, video=True)
    else:
        return render_template("results.html", max_count=MAX_COUNT, video=False)


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        # save video in dir
        f = request.files['file']

        global FILE
        FILE = f.filename
        video_path = os.path.join(os.getcwd(), 'uploads/' + FILE)
        f.save(video_path)

        # remove previous predictions
        files = glob.glob('static/images/*')
        for f in files:
            os.remove(f)

        # call method for object detection
        if request.form.get('video_check'):
            detect_units(video_path, video_check=True)
            return redirect(url_for('results', video_check=True))
        else:
            detect_units(video_path, video_check=False)
            return redirect(url_for('results', video_check=False))
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
