import matplotlib.pylab as plt

class Plotter:

    def __init__(self, data):

        self.points = []

        plt.ion()
        self.fig = plt.figure(1)
        self.ax1 = plt.subplot(2, 1, 1)

        self.fig.suptitle('Right click + z')
        self.ax1.plot(data[:, 0], 'b')
        self.ax2 = plt.subplot(2, 1, 2)
        self.ax2.plot(data[:, 1], 'r')

        self.fig.subplots_adjust(left=0.02, bottom=0.1, right=0.98, top=0.92, wspace=0.05, hspace=0.1)

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):

        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        # global points

        if event.key == 'z':
            self.points.append([int(event.xdata)])
            self.ax1.plot(event.xdata, event.ydata, 'o', color='r', markersize=5)
            self.ax1.axvline(x=event.xdata, color='r')
            self.ax2.plot(event.xdata, event.ydata, 'o', color='r', markersize=5)
            self.ax2.axvline(x=event.xdata, color='r')

            self.fig.canvas.draw()

class Plotter_2:

    def __init__(self, data_1, data_2):

        self.points = []

        plt.ion()
        self.fig = plt.figure(1)
        self.ax1 = plt.subplot(2, 1, 1)

        self.fig.suptitle('Right click + z for imu and x for mkr')
        self.ax1.plot( data_1, 'b')
        self.ax2 = plt.subplot(2, 1, 2)
        self.ax2.plot( data_2, 'r')

        self.fig.subplots_adjust(left=0.02, bottom=0.1, right=0.98, top=0.92, wspace=0.05, hspace=0.1)

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):

        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        # global points

        if event.key == 'z':
            self.points.append([int(event.xdata)])
            self.ax1.plot(event.xdata, event.ydata, 'o', color='b', markersize=5)
            self.ax1.axvline(x=event.xdata, color='b')


        if event.key == 'x':
            self.points.append([int(event.xdata)])
            self.ax2.plot(event.xdata, event.ydata, 'o', color='r', markersize=5)
            self.ax2.axvline(x=event.xdata, color='r')


            self.fig.canvas.draw()
