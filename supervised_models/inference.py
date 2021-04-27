import os
import sys

import cv2
from pympler.tracker import SummaryTracker
from tqdm import tqdm
import torch

sys.path.append(os.getcwd())

from supervised_models.train_sdd import SDDSegModel
from supervised_models.train_stanford import StanfordSegModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


def video_inference_by_still_frames(vid_path, checkpoint_path, n_frames_to_process=100, out_dir=None):
    """ Infer video segmentation using model trained on still frames (SDDSegModel)"""
    model = SDDSegModel().load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.setup("val")

    vid = cv2.VideoCapture(vid_path)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"video length: {length} res: {h}x{w} fps: {fps}")

    if out_dir is None:
        vid_writer = cv2.VideoWriter(vid_path.replace("video.mov", "out_effnet-b0-10.avi"),
                                     cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                     tuple(model.input_size[:2] * 2))
    else:
        vid_writer = cv2.VideoWriter(os.path.join(out_dir, vid_path.replace("/", "_").replace(".mov", ".avi")),
                                     cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                     tuple(model.input_size[:2] * 2))

    # preload frames

    for i_frame in tqdm(range(n_frames_to_process), "read & process"):
        ret, frame = vid.read()
        if frame is None:
            break
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, tuple(model.input_size[:2] * 2))
        out_im = model.forward_img(im)
        out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
        vid_writer.write(out_im)
    vid.release()
    vid_writer.release()


def video_inference_by_consecutive_frame_samples(model, vid_path, n_frames_to_process=100, smallest_size=640,
                                                 out_dir=None):
    vid = cv2.VideoCapture(vid_path)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"video length: {length} res: {h}x{w} fps: {fps}")

    # preload frames
    ims = []
    for i_frame in tqdm(range(n_frames_to_process), "reading"):
        ret, frame = vid.read()
        if frame is None:
            break
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # im = cv2.resize(im, tuple(model.input_size[:2] * 2), cv2.INTER_AREA)
        ims.append(im)
    vid.release()

    samples = []
    for i in range(n_frames_to_process):
        if i + 2 * model.interframe_step >= len(ims):
            break
        samples.append([ims[i], ims[i + model.interframe_step], ims[i + 2 * model.interframe_step]])
    output_frames = []
    for sample in tqdm(samples, "inference"):
        out_im = model.forward_img(sample, smallest_size=smallest_size)
        out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
        output_frames.append(out_im)

    print(output_frames[0].shape)
    if out_dir is None:
        vid_writer = cv2.VideoWriter(vid_path.replace("video.mov", "out_effnet-b0-10.avi"),
                                     cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                     (output_frames[0].shape[1], output_frames[0].shape[0]))  # * 2))
    else:
        vid_writer = cv2.VideoWriter(os.path.join(out_dir, vid_path.replace("/", "_").replace(".mov", ".avi")),
                                     cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                     (output_frames[0].shape[1], output_frames[0].shape[0]))  # * 2))
    for out_im in output_frames:
        vid_writer.write(out_im)
    vid_writer.release()


if __name__ == '__main__':
    n_frames_to_process = 1000
    smallest_size = 640
    out_dir = f"data/stanford_drone/out_{smallest_size}"
    os.makedirs(out_dir, exist_ok=True)
    # video_inference_by_still_frames("data/stanford_drone/videos/little/video0/video.mov",
    #                                 checkpoint_path='checkpoints_sdd/deeplabv3_effnet-b2/epoch=21-step=2639.ckpt',
    #                                 n_frames_to_process=n_frames_to_process,
    #                                 out_dir=out_dir)

    model = StanfordSegModel().load_from_checkpoint(
        "checkpoints_stanford/deeplabv3_effnet-b0-3ch/backup.epoch=22-step=15291.ckpt")
    model.to(device)
    model.setup("val")
    for i_vid, vid_dir in enumerate(model.val_dataset.video_dirs):
        tracker = SummaryTracker()

        print("---------------")
        print(i_vid, vid_dir)
        vid_path = os.path.join(vid_dir, "video.mov")
        video_inference_by_consecutive_frame_samples(model, vid_path, n_frames_to_process=n_frames_to_process,
                                                     smallest_size=smallest_size,
                                                     out_dir=out_dir)
        tracker.print_diff()
