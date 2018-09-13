# tensorflow-object-detection-Python-and-Cpp-
Some introduce and code for tensorflow object detection by using Python and C++

1.Ubuntu Environment prepare
  Protobuffer -- 3.5.1
  Tensorflow  -- r1.8
  bazel
  eigen -- 3.3
  
You can take following URL as a reference, but please use above library versions
https://github.com/hhzrz/tensorflow-cpp


2.Object detection model training and detect by python
For Chinese , you can take a look at following URL as reference
https://blog.csdn.net/chenmaolin88/article/details/79357263


2.0.Create a dataset in yolo like format:

'images' --> folder containing the training image
'labels' --> containing 1 annotation file for each image in .txt (one BB per row with class x-center y-center w h)
'traininglist.txt' --> a txt file where each row refer an image to be used as training sample, images and labels folder should be contained in the same directory
'validationlist.txt' --> a txt file where each row refer an image to be used as validation sample, images and labels folder should be contained in the same directory
'className.txt' --> a txt file with the name of the class to be displayed, one per row


2.1.yolo dataset format convert to tf record文件(include train and val record)
python python_tools/yolo_tf_converter.py -t data/train_images/traininglist.txt -o ../data/trainRecord -c ../data/className.txt
python python_tools/yolo_tf_converter.py -t data/val_images/validationlist.txt -o ../data/valRecord -c ../data/className.txt

2.2.Modify model.config
Modify num_classes
Modify PATH_TO_BE_CONFIGURED

2.3.Train model 
cd to tensorflow's path of object_detection

python train.py --logtostderr --pipeline_config_path=/home/zsb/Data/tf_data_L1_R_49VX80E_CN/output/model.config \
--train_dir=/home/zsb/Data/tf_data_L1_R_49VX80E_CN/saved_models/

2.4.output tensorflow object detection model
python export_inference_graph.py --pipeline_config_path=/home/zsb/Data/tensorflow_objectdetection/output/model.config --trained_checkpoint_prefix=/home/zsb/Data/tensorflow_objectdetection/saved_models/model.ckpt-15296  --output_directory=/home/zsb/Data/tensorflow_objectdetection/output/

2.5 python single image object detection
python tf_detect_img.py

2.6.python video object detection
python python_tools/video_detection.py -g /home/zsb/Data/tensorflow_objectdetection/output/pb_results/frozen_inference_graph.pb -l /home/zsb/Data/tensorflow_objectdetection/output/label_map.pbtxt -v /home/zsb/Data/Video/L1_R_49X7500F_CN/L1_R_49X75_CN_526-1.mp4

3.C++ object detection

Mostly copy from https://github.com/lysukhin/tensorflow-object-detection-cpp

I can build pass with above environment "1.Ubuntu Environment prepare".
And run video object detection correctly.

