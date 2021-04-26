import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pympler.tracker import SummaryTracker


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
            for i in range(row['x1'], row['x2'] + 1):
                assert 0 <= i < self.w, f"i={i} h={self.h} w={self.w} x2={row['x2']}"
                self.xs[i].append(ind)
            for j in range(row['y1'], row['y2'] + 1):
                assert 0 <= j < self.h, f"j={j} h={self.h} w={self.w} y2={row['y2']}"
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
        all_frame_ids_without_intersection = np.array(all_frame_ids_without_intersection)
        dists_to_cur_frame = np.abs(all_frame_ids_without_intersection - cur_frame_id)
        positions_by_dist = np.argsort(dists_to_cur_frame)
        positions_by_dist = positions_by_dist[:n_closest]
        closest_frame_ids = all_frame_ids_without_intersection[positions_by_dist]
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


def read_video(vid_path, n_box_max, n_frames_max):
    annot_df = read_vid_annotations(vid_path)  # read appropriate annotations
    print("num annotation bboxes:", len(annot_df))

    annot_df = annot_df.sort_values('frame')
    annot_df = annot_df.head(n_box_max)
    last_frame = annot_df["frame"].max()
    annot_df = annot_df[annot_df["frame"] != last_frame]
    if last_frame > n_frames_max:
        annot_df = annot_df[annot_df["frame"] <= n_frames_max]

    print("num annotation bboxes to process:", len(annot_df))

    vid = cv2.VideoCapture(vid_path)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w, h = int(w), int(h)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"video length: {length} res: {w}x{h} fps: {fps}")

    frames = []

    # preload frames
    while True:
        ret, frame = vid.read()
        if frame is None or len(frames) > max(annot_df["frame"]):
            break
        frames.append(frame)
    print("num frames to process", len(frames))
    vid.release()

    # print(h)
    # print(annot_df[annot_df['y2'] >= h].drop(columns=['occluded', "lost"]))
    # exit()
    annot_df.loc[annot_df['x1'] < 0, 'x1'] = 0
    annot_df.loc[annot_df['x2'] < 0, 'x2'] = 0
    annot_df.loc[annot_df['y1'] < 0, 'y1'] = 0
    annot_df.loc[annot_df['y2'] < 0, 'y2'] = 0

    annot_df.loc[annot_df['x1'] >= w, 'x1'] = w - 1
    annot_df.loc[annot_df['x2'] >= w, 'x2'] = w - 1
    annot_df.loc[annot_df['y1'] >= h, 'y1'] = h - 1
    annot_df.loc[annot_df['y2'] >= h, 'y2'] = h - 1

    return frames, annot_df, fps, w, h


def process_video(frames, annot_df, w, h, mask_thresh, needs_masked_img=False):
    video_rectangles = VideoRectangles(annot_df, w, h)

    output_masked_frames = []
    output_seg_masks = []
    output_box_masks = []
    output_frame_boxes = []

    class_2_label_id_map = {
        "Pedestrian": 1,
        "Biker": 2,
        "Bus": 3, "Car": 3, "Cart": 3,
        "Skater": 4,
    }

    for i_frame in tqdm(range(len(frames)), 'processing'):
        frame = frames[i_frame]
        orig_frame = frame[:, :, :]
        background_gray = cv2.cvtColor(frame[:, :, :], cv2.COLOR_BGR2GRAY)  # HSV)[:,:,2]

        frame_boxes = []
        bbox_class_mask = np.zeros(orig_frame.shape[:2])

        cur_df = annot_df[annot_df["frame"] == i_frame]
        for i, row in cur_df.iterrows():
            x1, y1, x2, y2 = row["x1"], row['y1'], row["x2"], row['y2']
            closest_frame_ids = video_rectangles.get_closest_frame_ids_without_intersection_with_box(i_frame, x1, y1,
                                                                                                     x2, y2,
                                                                                                     n_closest=4)
            try:
                bbox_class_mask[y1:y2, x1:x2] = class_2_label_id_map[row['label']]
            except Exception as e:
                print(bbox_class_mask.shape, y1, y2, x1, x2)
                print(row)
                raise e
            frame_boxes.append((i_frame, y1, y2, x1, x2, class_2_label_id_map[row['label']]))
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
        bbox_class_mask = bbox_class_mask.astype(np.uint8)

        classed_mask = (mask * bbox_class_mask).astype(np.uint8)
        output_seg_masks.append(classed_mask)
        output_box_masks.append(bbox_class_mask)
        output_frame_boxes.extend(frame_boxes)

        if needs_masked_img:
            classed_mask = (mask * bbox_class_mask).astype(np.uint8)
            segments = np.zeros_like(orig_frame)
            # print(np.unique(bbox_class_mask))
            # segments[:, :, 1] = mask * 255  # green colored mask
            segments[classed_mask == 1] = (255, 0, 0)  # car (blue)
            segments[classed_mask == 2] = (0, 255, 0)  # person (green)
            segments[classed_mask == 3] = (0, 255, 255)  # bicycle (yellow)

            res_img = cv2.addWeighted(orig_frame, 1, segments, 0.6, 1.0)

            for i, row in cur_df.iterrows():
                x1, y1, x2, y2 = row["x1"], row['y1'], row["x2"], row['y2']
                res_img = cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            output_masked_frames.append(res_img)

    return output_masked_frames, output_seg_masks, output_box_masks, output_frame_boxes


def save_video(vid_path, output_frames, fps, h, w):
    out_dir = "tmp/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, vid_path.replace("/", "_").replace("mov", "avi"))
    print(out_path, fps, (int(w), int(h)), len(output_frames), output_frames[0].shape)
    vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (int(h), int(w)))

    for frame in tqdm(output_frames, desc="writing"):
        vid_writer.write(frame)
        frame_ = frame
        if max(h, w) > 1200:
            frame_ = cv2.resize(frame, None, fx=0.6, fy=0.6)
        cv2.imshow(vid_path, frame_)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()

    vid_writer.release()


def show_and_save_video(video_dir, n_max, n_frames_max, mask_thresh):
    """
    Args:
         n_max: n first bboxes to process
         n_frames_max: n first frames to process
         mask_thresh: threshold for distinguish background from object (distance btw background and frame pixel)
    """

    vid_path = os.path.join(video_dir, "video.mov")
    frames, annot_df, fps, w, h = read_video(vid_path, n_max, n_frames_max)
    output_masked_frames, output_seg_masks, output_box_masks, output_frame_boxes = process_video(frames,
                                                                                                 annot_df, w, h,
                                                                                                 mask_thresh, True)
    save_video(vid_path, output_masked_frames, fps, h, w)


def store_as_dataset(video_dir, n_max, n_frames_max, mask_thresh):
    """
    Args:
         n_max: n first bboxes to process
         n_frames_max: n first frames to process
         mask_thresh: threshold for distinguish background from object (distance btw background and frame pixel)
    """

    vid_path = os.path.join(video_dir, "video.mov")
    frames, annot_df, fps, w, h = read_video(vid_path, n_max, n_frames_max)
    _, seg_masks, box_masks, frame_boxes = process_video(frames, annot_df, w, h, mask_thresh, False)

    boxes_df = pd.DataFrame(data=np.array(frame_boxes), columns=["frame", "x1", "y1", "x2", "y2", "label"])
    boxes_df.to_csv(os.path.join(video_dir, "boxes.csv"))
    os.makedirs(os.path.join(video_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(video_dir, "seg_masks"), exist_ok=True)
    os.makedirs(os.path.join(video_dir, "box_masks"), exist_ok=True)
    for i, (frame, seg_mask, box_mask, boxes) in tqdm(
            enumerate(zip(frames, seg_masks, box_masks, frame_boxes)), total=len(frames), desc="storing as dataset"):
        cv2.imwrite(os.path.join(video_dir, "frames", f"{i}.jpg"), frame)
        cv2.imwrite(os.path.join(video_dir, "seg_masks", f"{i}.png"), seg_mask)
        cv2.imwrite(os.path.join(video_dir, "box_masks", f"{i}.png"), box_mask)


def main():
    n_max = 10000  # n first bboxes to process
    n_frames_max = 3000
    mask_thresh = 35

    videos_dir = "data/stanford_drone/videos/"

    scene_names = ["bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad"]
    # val_videos_paths = [os.path.join(videos_dir, scene, "video0") for scene in scene_names]
    # show_and_save_video("data/stanford_drone/videos/coupa/video2", n_max, n_frames_max, mask_thresh)
    # exit(1)

    # to show [0!(hard dark low res), 1(car),4], 5! (car), 8! (large shadow), 11(car), 21!(good quality),26(fence), 31(solid persons), 36 (good quality),
    # final to show 1(car), 5! (car), 8! (large shadow), 11(car), 21!(good quality),26(fence), 31(solid persons), 36 (good quality)
    i_video = 0
    for scene in scene_names:
        scene_subdirs = os.listdir(os.path.join(videos_dir, scene))
        for vid_dir_name in scene_subdirs:
            tracker = SummaryTracker()

            vid_dir = os.path.join(videos_dir, scene, vid_dir_name)
            print("-----------------")
            print(i_video, vid_dir)
            # "data/stanford_drone/videos/coupa/video2"
            store_as_dataset(vid_dir, n_max, n_frames_max, mask_thresh)
            i_video += 1

            tracker.print_diff()


if __name__ == '__main__':
    main()
