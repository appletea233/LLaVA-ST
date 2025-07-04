### REC
```shell
python demo/demo.py \
--query "Please describe the location of sandwich in this image." \
--modality image \
--data_path demo/examples/rec/COCO_train2014_000000310450.jpg \
--show \
--task rec

python demo/demo.py \
--query "Please describe the location of a metal fork resting on a table in this image." \
--modality image \
--data_path demo/examples/rec/COCO_train2014_000000580631.jpg \
--show \
--task rec

python demo/demo.py \
--query "What is the specific area in this image that represents a person wearing a hat and a pink shirt?" \
--modality image \
--data_path demo/examples/rec/COCO_train2014_000000106755.jpg \
--show \
--task rec

python demo/demo.py \
--query "Where is catcher situated within this image?" \
--modality image \
--data_path demo/examples/rec/COCO_train2014_000000499679.jpg \
--show \
--task rec

python demo/demo.py \
--query "Where is bowl of carrots situated within this image?" \
--modality image \
--data_path demo/examples/rec/COCO_train2014_000000580957.jpg \
--show \
--task rec

```

### REG
```shell
python demo/demo.py \
--query "What object can be observed within the region [<w0.523>,<h0.621>,<w0.817>,<h0.937>] in the image?" \
--modality image \
--data_path demo/examples/reg/COCO_train2014_000000310289.jpg \
--show \
--task reg

python demo/demo.py \
--query "What is the primary object within the region [<w0.473>,<h0>,<w0.966>,<h0.982>]?" \
--modality image \
--data_path demo/examples/reg/COCO_train2014_000000508710.jpg \
--show \
--task reg

python demo/demo.py \
--query "What object can be observed within the region [<w0.33>,<h0.038>,<w0.904>,<h0.916>] in the image?" \
--modality image \
--data_path demo/examples/reg/COCO_train2014_000000368363.jpg \
--show \
--task reg

python demo/demo.py \
--query "What object can be observed within the region [<w0.3>,<h0.045>,<w0.724>,<h0.388>] in the image?" \
--modality image \
--data_path demo/examples/reg/COCO_train2014_000000144495.jpg \
--show \
--task reg
```

### TVG
```shell
python demo/demo.py \
--query "Give you a textual query: 'person takes a laptop from the shelf'. When does the described content occur in the video? Please return the start and end timestamps." \
--modality video \
--data_path demo/examples/tvg/0F7LW.mp4 \
--show \
--task tvg

python demo/demo.py \
--query "Give you a textual query: 'person dressing by putting a jacket on'. When does the described content occur in the video? Please return the start and end timestamps." \
--modality video \
--data_path demo/examples/tvg/5INX3.mp4 \
--show \
--task tvg
```

### TR
```shell
python demo/demo.py \
--query "Describe what took place during the period {<t0.409>,<t0.639>} in the video." \
--modality video \
--data_path demo/examples/tr/JYUqV-7BPWE.mp4 \
--show \
--task tr

python demo/demo.py \
--query "What was going on within the time frame of {<t0.289>,<t0.509>}  in the video?" \
--modality video \
--data_path demo/examples/tr/5PbDHfKFNQs.mp4 \
--show \
--task tr
```
### STVG
```shell
python demo/demo.py \
--query "At which time interval in the video can we see a child wearing a red hat holds a bat occurring? Please describe the location of the corresponding subject/object in this video.Please firstly give the timestamps, and then give the spatial bounding box corresponding to each timestamp in the time period." \
--modality video \
--data_path demo/examples/stvg/8665030691.mp4 \
--show \
--task stvg \
--split "[0, 829]"

python demo/demo.py \
--query "At which time interval in the video can we see an adult in white clothes watches a fastest adult occurring? Please describe the location of the corresponding subject/object in this video.Please firstly give the timestamps, and then give the spatial bounding box corresponding to each timestamp in the time period." \
--modality video \
--data_path demo/examples/stvg/5875096370.mp4 \
--show \
--task stvg \
--split "[901, 1221]"
```
### SVG
```shell

python demo/demo.py \
--query "Between {<t0.087>,<t0.581>}, who does the adult in white clothes watch? Please describe the location of the corresponding subject/object in this video.Please give the spatial bounding box corresponding to each timestamp in the time period." \
--modality video \
--data_path demo/examples/svg/5875096370.mp4 \
--show \
--task stvg \
--split "[901, 1221]"

python demo/demo.py \
--query "In the time range {<t0.247>,<t0.335>}, a child grabs a ball in the playground. Can you identify the position of the corresponding subject/object within this video?Please give the spatial bounding box corresponding to each timestamp in the time period." \
--modality video \
--data_path demo/examples/svg/4591061010.mp4 \
--show \
--task stvg \
--split "[346, 743]"

```

### ELC
```shell
python demo/demo.py \
--query "Starts in <t0.591>, what happened about the subject/object within the specified region [<w0.352>, <h0.442>, <w0.505>, <h0.827>]? Where is the corresponding subject/object located?Please firstly give the end timestamp, then give the event associated with the object/subject, finally give the spatial bounding box corresponding to each timestamp in the time period." \
--modality video \
--data_path demo/examples/elc/4240545211.mp4 \
--show \
--task stvg \
--split "[0, 257]"

python demo/demo.py \
--query "Starts in <t0.79>, describe the event about subject/object located within the region [<w0.009>, <h0.102>, <w0.302>, <h0.819>]. Please describe the location of the corresponding subject/object in this video.Please firstly give the end timestamp, then give the event associated with the object/subject, finally give the spatial bounding box corresponding to each timestamp in the time period." \
--modality video \
--data_path demo/examples/elc/8115037154.mp4 \
--show \
--task stvg \
--split "[0, 1379]"
```

### dgc
```shell
python demo/demo.py \
--query "Can you provide a thorough description of this image? Please output with interleaved bounding boxes for the corresponding phrases." \
--modality image \
--data_path demo/examples/dgc/sa_10036163.jpg \
--show \
--task dgc 

python demo/demo.py \
--query "Could you give me an elaborate explanation of this picture? Please respond with interleaved bounding boxes for the corresponding phrases." \
--modality image \
--data_path demo/examples/dgc/sa_10036241.jpg \
--show \
--task dgc 

python demo/demo.py \
--query "Could you give me an elaborate explanation of this picture? Please respond with interleaved bounding boxes for the corresponding phrases." \
--modality image \
--data_path demo/examples/dgc/sa_10061427.jpg \
--show \
--task dgc 

python demo/demo.py \
--query "Please describe in detail the contents of the image. Please respond with interleaved bounding boxes for the corresponding parts of the answer." \
--modality image \
--data_path demo/examples/dgc/sa_10061425.jpg \
--show \
--task dgc 

python demo/demo.py \
--query "Please describe in detail the contents of the image. Please respond with interleaved bounding boxes for the corresponding parts of the answer." \
--modality image \
--data_path demo/examples/dgc/sa_10037708.jpg \
--show \
--task dgc 
```
