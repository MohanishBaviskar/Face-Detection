# Real-Time Face Detection: Haar Cascade vs a VGG16 Detector

Two ways to find and box faces in a live webcam feed: a classical OpenCV Haar cascade, and a deep-learning detector
built on VGG16 and trained end to end on a small, self-collected and self-labelled dataset.

> This project *detects and localises* faces (where is a face?). It does not identify whose face it is.

## 1. Haar cascade (`haarcascade_face_detection.py`)

OpenCV's pre-trained frontal-face cascade (`haarcascade_frontalface_alt.xml`) runs on each grayscale frame
(`detectMultiScale`, scale factor 1.1, 5 neighbours) and draws a box around every face found.

- No training needed, runs in real time on a CPU.
- Sensitive to head rotation and lighting, since it uses hand-crafted features.

## 2. VGG16 face detector (`vgg16_face_detector.ipynb`)

A full pipeline from data collection to live inference:

1. **Collect data:** 30 webcam images captured with OpenCV.
2. **Label:** draw face bounding boxes with [labelme](https://github.com/wkentaro/labelme); split 70 / 15 / 15
   into train, test and validation.
3. **Augment:** [Albumentations](https://albumentations.ai/) creates 60 variants per image (random crop, flips,
   brightness / contrast, gamma, RGB shift), with the bounding boxes transformed accordingly.
4. **Model:** a VGG16 backbone (ImageNet weights, input 120×120) with two heads:
   - **classification:** is there a face? (sigmoid)
   - **regression:** the box's 4 corner coordinates (sigmoid, normalised)
5. **Loss:** binary cross-entropy for the class, plus a custom localisation loss (squared error on the box's
   top-left corner and on its width and height); total = localisation + 0.5 × classification.
6. **Training:** a custom `tf.keras.Model` subclass with its own `train_step` and `test_step`, Adam (lr 1e-4 with
   decay), 10 epochs, TensorBoard logging.
7. **Live inference:** each webcam frame is resized to 120×120, and a box with a "face" label is drawn when the
   predicted face probability is above 0.5.

## Run it

```bash
pip install -r requirements.txt

# Haar cascade, live webcam (press q to quit)
python haarcascade_face_detection.py

# VGG16 detector
jupyter notebook vgg16_face_detector.ipynb
```

Notes for the notebook:
- The images, labels and trained model are not included. Run the notebook's collection and labelling steps first to
  build your own dataset.
- Paths in the notebook are Windows-style (`C:\code\data`); change `IMAGES_PATH` and the dataset paths to your own
  folder.
- The live-inference cell uses camera index 1; change `cv2.VideoCapture(1)` to `0` for a built-in webcam.

**Tools:** Python, OpenCV, TensorFlow / Keras 2.14, VGG16 transfer learning, Albumentations, labelme.
