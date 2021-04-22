import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_vid_annotations(vid_path: str) -> pd.DataFrame:
    annotation_path = vid_path.replace("videos", "annotations")
    annotation_path = annotation_path.replace('video.mov', "annotations.txt")
    df = None
    if os.path.exists(annotation_path):
        df = pd.read_csv(annotation_path, sep=" ", header=None,
                         names=["track", 'x1', 'y1', 'x2', 'y2', 'frame', 'lost', 'occluded', 'generated', 'label'])
    df = df[df["lost"] != 1]
    df = df[df["occluded"] != 1]

    return df


def read_video(vid_path, n_max):
    print(vid_path)

    annot_df = read_vid_annotations(vid_path)  # read appropriate annotations
    print("num annotation bboxes:", len(annot_df))

    annot_df = annot_df.sort_values('frame')
    annot_df = annot_df.head(n_max)
    last_frame = annot_df["frame"].max()
    annot_df = annot_df[annot_df["frame"] != last_frame]
    print("num annotation bboxes to process:", len(annot_df))

    vid = cv2.VideoCapture(vid_path)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"video length: {length} res: {h}x{w} fps: {fps}")

    frames = []

    # preload frames
    while (1):
        ret, frame = vid.read()
        if frame is None or len(frames) > max(annot_df["frame"]):
            break
        frames.append(frame)
    print("num frames to process", len(frames))
    vid.release()

    return frames, annot_df, fps, h, w


def process_video(vid_path, frames, annot_df, h, w, mask_thresh):
    output_frames = []
    for i_frame in tqdm(range(len(frames)), 'processing'):
        frame = frames[i_frame]
        orig_frame = frame[:, :, :]
        frame_gray = cv2.cvtColor(frame[:, :, :], cv2.COLOR_BGR2GRAY)  # HSV)[:,:,2]

        magnitudes = []
        # other_frame_ixs = [-8, -4, -2, -1, 1, 2, 4, 8]
        other_frame_ixs = [-4, 2]
        for other_frame_ix in other_frame_ixs:
            i_other = i_frame - other_frame_ix
            if not 0 <= i_other < len(frames):
                continue
            frame_other = frames[i_other]
            other_gray = cv2.cvtColor(frame_other, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(frame_gray, other_gray, None, 0.5, 3, 10, 3, 5, 1.2, 0)
            # flow = cv2.optflow.calcOpticalFlowSparseToDense(frame2_gray, frame2_gray)

            flow_mag, flow_ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag = cv2.normalize(flow_mag, None, 0, 255, cv2.NORM_MINMAX)
            magnitudes.append(flow_mag)
            assert frame_gray.shape == flow_mag.shape, f"{frame_gray.shape} {flow_mag.shape}"

        mask = np.mean(magnitudes, axis=0).astype(dtype=np.uint8)

        cur_df = annot_df[annot_df["frame"] == i_frame]
        mask_zeros = np.zeros_like(mask)
        for i, row in cur_df.iterrows():
            x1, y1, x2, y2 = row["x1"], row['y1'], row["x2"], row['y2']
            mask_zeros[y1:y2, x1:x2] = 1
        mask = mask * mask_zeros
        # print("mask", mask.dtype, np.min(mask), np.max(mask), np.sum(mask>10),"/",np.sum(mask>=0))
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        mask[mask <= mask_thresh] = 0

        mask[mask > 0] = 1
        mask = mask.astype(np.uint8)
        segments = np.zeros_like(orig_frame)
        segments[:, :, 1] = mask * 255  # green colored mask

        res_img = cv2.addWeighted(orig_frame, 0.8, segments, 0.4, 1.0)

        for i, row in cur_df.iterrows():
            x1, y1, x2, y2 = row["x1"], row['y1'], row["x2"], row['y2']
            res_img = cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        output_frames.append(res_img)
        if max(h, w) > 1700:
            res_img = cv2.resize(res_img, None, fx=0.5, fy=0.5)
        cv2.imshow(vid_path, res_img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
    return output_frames


def save_video(vid_path, output_frames, fps, h, w):
    out_dir = "data/output/vanilla_background"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, vid_path.replace("/", "_").replace("mov", "avi"))
    print(out_path, fps, (int(w), int(h)), len(output_frames), output_frames[0].shape)
    vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (int(h), int(w)))

    for frame in tqdm(output_frames, desc="writing"):
        vid_writer.write(frame)

    vid_writer.release()


def main():
    vid_dir = "data/videos"

    scene_names = ["bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad"]
    val_videos_paths = [os.path.join(vid_dir, scene, "video0/video.mov") for scene in scene_names]
    n_max = 10000  # n first bboxes to process
    mask_thresh = 25

    for vid_path in val_videos_paths:
        frames, annot_df, fps, h, w = read_video(vid_path, n_max)
        output_frames = process_video(vid_path, frames, annot_df, h, w, mask_thresh)
        save_video(vid_path, output_frames, fps, h, w)


if __name__ == '__main__':
    main()
