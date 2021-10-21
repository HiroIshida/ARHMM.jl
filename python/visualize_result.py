import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from PIL import Image
from moviepy.editor import ImageSequenceClip

def canvas_to_ndarray(fig, resize_pixel=None):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if resize_pixel is None:
        return data
    img = Image.fromarray(data)
    img_resized = img.resize(resize_pixel)
    data_resized = np.asarray(img_resized)
    return data_resized

if __name__=='__main__':
    cache_dir = os.path.expanduser('~/.kyozi')
    with open(os.path.join(cache_dir, 'summary_chunk.pickle'), 'rb') as f:
        chunk = pickle.load(f)
    with open(os.path.join(cache_dir, 'arhmm_result.pickle'), 'rb') as f:
        arhmm_result = pickle.load(f)

    savepath = osp.join(cache_dir, 'seg_debug_video')
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for i, (img_seq, cmd_seq, phase_seq) in enumerate(
            zip(chunk['img_seqs'], chunk['cmd_seqs'], arhmm_result)):
        print("processing epoch: {}".format(i))
        fig, ax = plt.subplots()
        imgs_matplotlib = []
        im_handle = None
        tx_handle = None
        colors = ['blue', 'green', 'red', 'yellow']
        props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
        for img, phase in zip(img_seq, phase_seq):
            text = "phase: {}".format(phase)
            color = colors[phase]
            if im_handle is None:
                im_handle = ax.imshow(img)
                tx_handle = ax.text(
                        7, 1, text, fontsize=25, color=color, bbox=props, verticalalignment='top')
            else:
                im_handle.set_data(img)
                tx_handle.set_text(text)
                tx_handle.set_color(color)
            fig.canvas.draw()
            fig.canvas.flush_events()
            imgs_matplotlib.append(canvas_to_ndarray(fig))
        filename = osp.join(savepath, 'seq{}.gif'.format(i))
        clip = ImageSequenceClip(list(imgs_matplotlib), fps=10)
        clip.write_gif(filename, fps=10)
