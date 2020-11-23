
# shoes

- yolov4 for detecting shoes object

**[required]**
1. use [`tensorflow-yolov4-tflite`](https://github.com/hunglc007/tensorflow-yolov4-tflite)
2. convert the trained weights to tensorflow checkpoint.


## 

```sh
# run docker
$ docker run --rm  --runtime=nvidia -it -p 9999:9999 -v $(pwd)/trainings:/training/custom_training -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro my-shoes:tflite /bin/bash

#in docker
cd ~/training
cp ../my-shoes_20201120_23\:04\:28/weights/yolov4_best.weights ./data/yolov4.weights
# convert weight -> checkpoint
py save_model.py --input_size 416 --model yolov4
# check detect
py detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/shoes.jpg
# run simple-api
uvicorn api:app --host 0.0.0.0 --port 9999
```
