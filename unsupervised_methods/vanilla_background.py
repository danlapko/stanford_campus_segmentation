import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class VideoRectangles:
    def __init__(self, annot_df: pd.DataFrame, w, h):
        self.h = int(h)
        self.w = int(w)
        self.annot_df = annot_df
        self.n_frames = max(annot_df['frame'])
        self.xs = [[] for _ in range(self.w)]  # contains list of bbox ids for each x-axis pixel
        self.ys = [[] for _ in range(self.h)]  # contains list of bbox ids for each y-axis pixel
        self._full_fill_xs_n_ys()

    def _full_fill_xs_n_ys(self):
        for ind, row in self.annot_df.iterrows():
            for i in range(row['x1'], row['x2']):
                self.xs[i].append(ind)
            for j in range(row['y1'], row['y2']):
                self.ys[j].append(ind)

    def get_rect_by_id(self, rect_id: int):
        return self.annot_df.loc[rect_id]

    def get_rects_by_ids(self, rect_ids: int):
        return self.annot_df.loc[rect_ids]

    def get_rect_ids_for_box(self, x1, y1, x2, y2):
        x_rects = set()
        for x in range(x1, x2):
            x_rects.update(self.xs[x])

        y_rects = set()
        for y in range(y1, y2):
            y_rects.update(self.ys[y])

        return x_rects & y_rects

    def get_frame_ids_with_intersection_with_box(self, x1, y1, x2, y2):
        intersected_rect_ids = self.get_rect_ids_for_box(x1, y1, x2, y2)
        tmp_df = self.annot_df[self.annot_df.index.isin(intersected_rect_ids)]
        intersected_frame_ids = tmp_df["frame"].unique()
        return intersected_frame_ids

    def get_frame_ids_without_intersection_with_box(self, x1, y1, x2, y2):
        intersected_frame_ids = self.get_frame_ids_with_intersection_with_box(x1, y1, x2, y2)
        all_frame_ids = self.annot_df["frame"].unique()

        return list(set(all_frame_ids) - set(intersected_frame_ids))

    def get_closest_frame_ids_without_intersection_with_box(self, cur_frame_id, x1, y1, x2, y2, n_closest=1):
        all_frame_ids_without_intersection = self.get_frame_ids_without_intersection_with_box(x1, y1, x2, y2)
        # min_dist = self.n_frames
        # best_frame_id = None
        all_frame_ids_without_intersection = np.array(all_frame_ids_without_intersection)
        dists_to_cur_frame = np.abs(all_frame_ids_without_intersection - cur_frame_id)
        positions_by_dist = np.argsort(dists_to_cur_frame)
        positions_by_dist = positions_by_dist[:n_closest]
        closest_frame_ids = all_frame_ids_without_intersection[positions_by_dist]
        # for other_frame_id in all_frame_ids_without_intersection:
        #     if abs(cur_frame_id - other_frame_id) < min_dist:
        #         min_dist = abs(cur_frame_id - other_frame_id)
        #     best_frame_id = other_frame_id
        return closest_frame_ids.tolist()


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
    video_rectangles = VideoRectangles(annot_df, h, w)

    output_frames = []
    for i_frame in tqdm(range(len(frames)), 'processing'):
        frame = frames[i_frame]
        orig_frame = frame[:, :, :]
        background_gray = cv2.cvtColor(frame[:, :, :], cv2.COLOR_BGR2GRAY)  # HSV)[:,:,2]

        cur_df = annot_df[annot_df["frame"] == i_frame]
        for i, row in cur_df.iterrows():
            x1, y1, x2, y2 = row["x1"], row['y1'], row["x2"], row['y2']
            closest_frame_ids = video_rectangles.get_closest_frame_ids_without_intersection_with_box(i_frame, x1, y1,
                                                                                                     x2, y2,
                                                                                                     n_closest=4)
            background_crop_grays = []
            for closest_frame_id in closest_frame_ids:
                closest_frame = frames[closest_frame_id]
                closest_frame_gray = cv2.cvtColor(closest_frame, cv2.COLOR_BGR2GRAY)  # HSV)[:,:,2]
                background_crop_gray = closest_frame_gray[y1:y2, x1:x2]
                background_crop_grays.append(background_crop_gray.astype(np.int32))
            if len(background_crop_grays) > 0:
                background_gray[y1:y2, x1:x2] = np.mean(background_crop_grays, axis=0).astype(dtype=np.uint8)

        mask = abs(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY).astype(np.int32) - background_gray.astype(np.int32))
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
            res_img = cv2.resize(res_img, None, fx=0.6, fy=0.6)
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
    # to show [0!(hard dark low res), 1(car),4], 5! (car), 8! (large shadow), 11(car), 21!(good quality),26(fence), 31(solid persons), 36 (good quality),
    # final to show 1(car), 5! (car), 8! (large shadow), 11(car), 21!(good quality),26(fence), 31(solid persons), 36 (good quality)
    # vid_id = 26  # video num to process
    n_max = 10000  # n first bboxes to process
    mask_thresh = 35

    vid_path = val_videos_paths[3]
    frames, annot_df, fps, h, w = read_video(vid_path, n_max)
    output_frames = process_video(vid_path, frames, annot_df, h, w, mask_thresh)
    save_video(vid_path, output_frames, fps, h, w)


if __name__ == '__main__':
    main()
