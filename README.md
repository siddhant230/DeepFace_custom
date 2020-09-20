# DeepFace_custom
custom library for deepface

1. Comes with better and more reliable face extractor, instead of Haar-cascade xmls; that are prone to changes.
2. OOP approach to load the model once, and use it repeatedly as per the task.
3. Current support of only facenet is given. If you want to add any other model like VGG or Deepface along with architectural py file,
    then just download the weight in weights folder and change the input layer resize factor in DeepFace_custom.
