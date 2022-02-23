import matplotlib

matplotlib.use("TkAgg")
import time
from matplotlib import pyplot as plt
import numpy as np


def live_update_demo(blit=False):

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    text = ax2.text(0.8, 0.5, "")


    axnum = [ax1, ax2, ax3, ax4]
    for i in axnum:
        i.set_xlim(0, 1000)
        i.set_ylim([0, 2000])
        i.set_title('Kick Velocity')
    lines = []
    for i in axnum:
        line, = i.plot([],[], lw=2)
        lines.append(line)



    fig.canvas.draw()  # note that the first draw comes before setting data

    if blit:
        # cache the background
        backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axnum]

    plt.show(block=False)

    t_start = time.time()
    k = 0.
    xx = []
    yy = []
    for i in np.arange(1000):
        xx.append(i)
        yy.append(i*2)
        for j in lines:
            j.set_data(xx, yy)
        # print tx
        k += 0.11
        if blit:
            # restore background
            # fig.canvas.restore_region(axbackground)
            # fig.canvas.restore_region(ax2background)
            for j in backgrounds:
                fig.canvas.restore_region(j)
            # redraw just the points
            # ax1.draw_artist(img)
            for j in range(len(axnum)):
                axnum[j].draw_artist(lines[j])
            # ax2.draw_artist(line)
            axnum[1].draw_artist(text)

            # fill in the axes rectangle
            # fig.canvas.blit(ax1.bbox)
            # fig.canvas.blit(ax2.bbox)
            for j in axnum:
                fig.canvas.blit(j.bbox)

            # in this post http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
            # it is mentionned that blit causes strong memory leakage.
            # however, I did not observe that.

        else:
            # redraw everything
            fig.canvas.draw()

        fig.canvas.flush_events()
        # alternatively you could use
        # plt.pause(0.000000000001)
        # however plt.pause calls canvas.draw(), as can be read here:
        # http://bastibe.de/2013-05-30-speeding-up-matplotlib.html


live_update_demo(True)  # 175 fps
# live_update_demo(False) # 28 fps
