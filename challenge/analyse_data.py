from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

figure = go.FigureWidget(make_subplots(rows=8, cols=1))

for i, signal in tqdm(enumerate(signals)):
    signal_data = np.random.choice(data[:, 2+9000*i:2+9000*(i+1)].flatten(), int(1e6))

    figure.add_trace(go.Histogram(x=signal_data), row=i+1, col=1)
    histogram = figure.data[-1]

    def adjust_histogram_data(xaxis, xrange):
        x_values_subset = signal_data[np.logical_and(xrange[0] <= signal_data,
                                                  signal_data <= xrange[1])]
        histogram.x = x_values_subset
    figure.layout.xaxis.on_change(adjust_histogram_data, 'range')
    figure.update_layout(bargap=0.0)

figure.show()


import h5py

starts, durations = [], []
for i in range(22):
    with h5py.File(f"/gpfs/users/idrissib/datasets/sleepapnea/records/train/{i}.h5") as f:
        starts.append(f["apnea/start"][:])
        durations.append(f["apnea/duration"][:])

all_starts, all_durations = np.concatenate(starts), np.concatenate(durations)

fig = go.Figure(data=go.Histogram(x=all_starts))

fig.write_html("figures/apnea_starts.html")

fig = go.Figure(data=go.Histogram(x=all_durations))

fig.write_html("figures/apnea_durations.html")