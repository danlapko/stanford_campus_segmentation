### This repo contains semantic segmentation models for Stanford Drone Dataset and for Semantic Drone Dataset

Detailed solution including visualization results and links to model weights you can find in [solution.md](solution.md).

Stanford Drone Dataset https://cvgl.stanford.edu/projects/uav_data/   
Semantic Drone Dataset https://www.tugraz.at/index.php?id=22387

#### Stanford Drone Dataset

1. download dataset;
2. unpack into `data/stanford_drone` folder, it should contain `annotations/` and `videos/` subfolders;
3. run `python unsupervised_methods/vanilla_background.py` to prepare segmentation masks based on background subtraction;
4. each video folder now should contain `box_masks/`, `frames/`, `seg_masks/` subfolders and `boxes.csv`;
5. run `python supervised_models/train_stanford.py` for training;
6. run `tensorboard --logdir=lightning_logs/version_0` to see logs;
7. run `python supervised_models/inference.py` for inference on validation set. Don't forget to point appropriate checkpoint inside inference.py;

#### Semantic Drone Dataset

1. download dataset (https://www.kaggle.com/bulentsiyah/semantic-drone-dataset);
2. unpack into `data/SDD` folder, it should contain `RGB_color_image_masks/`,  `semantic_drone_dataset/` subfolders and `class_dict_seg.csv`;
3. run `python supervised_models/sdd_dataset.py` to prepare 1-channel (not colored) segmentation masks, categories encoded as int labels: `{0:"default", 1:"car", 2:"person", 3:"bicycle"}`;
4. `data/SDD` now should contain `value_masks` subfolder with value masks for each dataset image;
5. run `python supervised_models/train_sdd.py` for training;
6. run `tensorboard --logdir=lightning_logs/version_0` to see logs;
7. run `python supervised_models/inference.py` for inference on validation set. Don't forget to point appropriate checkpoint and model inside inference.py;