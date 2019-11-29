import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def show(transform, data):
    '''
        size of transform : (n, 2)
        size of data : (n, 784)
        '''
    fig, ax = plt.subplots()
    ax.scatter(transform[:, 0], transform[:, 1])
    for x0, y0, img in zip(transform[:, 0], transform[:, 1], data.reshape((-1, 28, 28), order='F')):
        ab = AnnotationBbox(OffsetImage(img, zoom=0.4, cmap='gray'), (x0, y0), frameon=False)
        ax.add_artist(ab)
    plt.show()

