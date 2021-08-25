import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

g = 9.8                                                        #value of gravity
v = 10.0                                                       #initial velocity
theta = 40.0 * np.pi / 180.0                                   #initial angle of launch in radians
t = 2 * v * np.sin(theta) / g
t = np.arange(0, 0.1, 0.01)                                    #time of flight into an array
x = np.arange(0, 0.1, 0.01)
line, = ax.plot(x, v * np.sin(theta) * x - (0.5) * g * x**2)   # plot of x and y in time

def animate(i):
    """change the divisor of i to get a faster (but less precise) animation """
    line.set_xdata(v * np.cos(theta) * (t + i /10.0))
    line.set_ydata(v * np.sin(theta) * (x + i /10.0) - (0.5) * g * (x + i / 10.0)**2)
    return line,

plt.axis([0.0, 10.0, 0.0, 5.0])
ax.set_autoscale_on(False)

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200))


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# References
# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
# https://riptutorial.com/matplotlib/example/23558/basic-animation-with-funcanimation

def func(t, line):
    t = np.arange(0, t, 0.1)
    y = np.sin(t)
    line.set_data(t, y)
    return line


fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(-1.2, 1.22))
redDots = plt.plot([], [], 'ro')
line = plt.plot([], [], lw=2)

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, func, frames=np.arange(1, 100, 0.1), fargs=(line), interval=100, blit=False)
# line_ani.save(r'Animation.mp4')



import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import pandas as pd
from sys import exit
def update_lines(num, data, line):
    # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(data[0:2, :num])
    line.set_3d_properties(data[2, :num])
    return line

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Reading the data from a CSV file using pandas
repo = pd.read_csv('hist.csv', header= None)
repo.columns=['x', 'y']
repo.to_csv('hist.csv', index=False) # save to new csv file
data = np.array((repo['x'].values, repo['y'].values))
print(data.shape[1])
exit()

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
limit = 100000.
line = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])[0]

# Setting the axes properties
ax.set_xlim3d([-limit, limit])
ax.set_xlabel('X')

ax.set_ylim3d([-limit, limit])
ax.set_ylabel('Y')

ax.set_zlim3d([-limit, limit])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, data.shape[1], fargs=(data, line), interval=50, blit=False)

plt.show()

"""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=10)

def animate_traffic():
    fig=plt.figure(1)
    ax=fig.add_subplot(1,1,1)
    tsim=tstart
    with writer.saving(fig, "roadtest.mp4", 100):
        for i in range(100):
            draw_roadlayout()
            for car in cars:
                # draw each of the cars on the road
                # based on time tsim
            plt.grid(False)
            ax.axis(plt_area)
            fig   = plt.gcf()
            writer.grab_frame()
            ax.cla()
            tsim+=timestep
    plt.close(1) """