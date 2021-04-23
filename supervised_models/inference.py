import cv2
from tqdm import tqdm
import torch

from supervised_models.train_sdd import SDDSegModel

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


if __name__ == '__main__':
    video_inference_by_still_frames("data/stanford_drone/videos/little/video0/video.mov")
