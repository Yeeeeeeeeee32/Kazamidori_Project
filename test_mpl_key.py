import matplotlib.pyplot as plt
def on_click(event):
    print(f"Click! key={event.key}, x={event.xdata}, y={event.ydata}")
fig, ax = plt.subplots()
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
