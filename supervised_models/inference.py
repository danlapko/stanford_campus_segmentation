import os

import cv2
from tqdm import tqdm
import torch

from supervised_models.train_sdd import SDDSegModel
from supervised_models.train_stanford import StanfordSegModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def video_inference_by_still_frames(vid_path):
    model = SDDSegModel().load_from_checkpoint('checkpoints/deeplabv3_effnet-b2/epoch=21-step=2639.ckpt')
    model.to(device)
    model.setup("val")
    n_frames_to_process = 100
    print(vid_path)

    vid = cv2.VideoCapture(vid_path)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"video length: {length} res: {h}x{w} fps: {fps}")

    vid_writer = cv2.VideoWriter(vid_path.replace("video.mov", "out.avi"), cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                 tuple(model.input_size[:2] * 2))

    # preload frames

    for i_frame in tqdm(range(n_frames_to_process)):
        ret, frame = vid.read()
        if frame is None:
            break
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, tuple(model.input_size[:2] * 2))
        out_im = model.forward_img(im)
        out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
        # cv2.imshow(vid_path, out_im)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
        vid_writer.write(out_im)
    vid.release()
    vid_writer.release()
    # cv2.destroyAllWindows()


def video_inference_by_triplets(model, vid_path, n_frames_to_process=100, out_dir=None):
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
    ims = []
    for i_frame in tqdm(range(n_frames_to_process)):
        ret, frame = vid.read()
        if frame is None:
            break
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, tuple(model.input_size[:2] * 2), cv2.INTER_AREA)
        ims.append(im)
    vid.release()

    triplets = []
    for i in range(n_frames_to_process):
        if i + 2 * model.interframe_step >= len(ims):
            break
        triplets.append([ims[i], ims[i + model.interframe_step], ims[i + 2 * model.interframe_step]])
    for triplet in tqdm(triplets, "inference"):
        out_im = model.forward_img(triplet)
        out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
        # cv2.imshow(vid_path, out_im)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
        vid_writer.write(out_im)
    vid_writer.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    # video_inference_by_still_frames("data/stanford_drone/videos/little/video0/video.mov")

    model = StanfordSegModel().load_from_checkpoint(
        "checkpoints_stanford/deeplabv3_effnet-b2-3ch/epoch=22-step=15291.ckpt")
    model.to(device)
    model.setup("val")
    # video_inference_by_triplets(model, "data/stanford_drone/videos/little/video0/video.mov")
    out_dir = "data/stanford_drone/out"
    os.makedirs(out_dir, exist_ok=True)
    for i_vid, vid_dir in enumerate(model.val_dataset.video_dirs):
        print("---------------")
        print(i_vid, vid_dir)
        vid_path = os.path.join(vid_dir, "video.mov")
        video_inference_by_triplets(model, vid_path, n_frames_to_process=1000, out_dir=out_dir)
