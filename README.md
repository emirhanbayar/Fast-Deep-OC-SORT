# Fast-Deep-OC-SORT


## Installation

As in Deep-OC-SORT, follow the following instructions.  

After cloning, install external dependencies: 
```
cd external/YOLOX/
pip install -r requirements.txt && python setup.py develop
cd ../deep-person-reid/
pip install -r requirements.txt && python setup.py develop
cd ../fast_reid/
pip install -r docs/requirements.txt
```

OCSORT dependencies are included in the external dependencies. If you're unable to install `faiss-gpu` needed by `fast_reid`, 
`faiss-cpu` should be adequate. Check the external READMEs for any installation issues.

Add [the weights](https://drive.google.com/drive/folders/1cCOx_fadIOmeU4XRrHgQ_B5D7tEwJOPx?usp=sharing) to the 
`external/weights` directory (do NOT untar the `.pth.tar` YOLOX files).

## Data

Place MOT17/20 and DanceTrack under:

```
data
|——————mot (this is MOT17)
|        └——————train
|        └——————test
|——————MOT20
|        └——————train
|        └——————test
|——————dancetrack
|        └——————train
|        └——————test
|        └——————val
```

and run:

```
python3 data/tools/convert_mot17_to_coco.py
python3 data/tools/convert_mot20_to_coco.py
python3 data/tools/convert_dance_to_coco.py
```

## Tracking

For Deep-OC-SORT, which is the baseline, run:

```bash 
exp=best_paper_ablations
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset mot17 --w_assoc_emb 0.75 --aw_param 0.5
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset mot20 --track_thresh 0.4 --w_assoc_emb 0.75 --aw_param 0.5
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset dance --aspect_ratio_thresh 1000 --w_assoc_emb 1.25 --aw_param 1
```

For Fast-Deep-OC-SORT add the following flags:

```bash
exp=best_paper_ablations
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset mot17 --w_assoc_emb 0.75 --aw_param 0.5 --occlusion_threshold {IoU_threshold} --aspect_ratio_threshold {aspect_ratio_threshold}
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset mot20 --track_thresh 0.4 --w_assoc_emb 0.75 --aw_param 0.5 --occlusion_threshold {IoU_threshold} --aspect_ratio_threshold {aspect_ratio_threshold}
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset dance --aspect_ratio_thresh 1000 --w_assoc_emb 1.25 --aw_param 1 --occlusion_threshold {IoU_threshold} --aspect_ratio_threshold {aspect_ratio_threshold}
```

where `{IoU_threshold}` and `{aspect_ratio_threshold}` are the parameters that are introduced in Fast-Deep-OC-SORT, and explained in the paper.


## Evaluation

To run TrackEval for HOTA and Identity with linear post-processing on MOT17, run:

```bash
python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA Identity \
  --TRACKERS_TO_EVAL ${exp}_post \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK MOT17
```