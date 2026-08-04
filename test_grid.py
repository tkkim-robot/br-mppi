import matplotlib.pyplot as plt

num_algos = 7
algos = ["brmppi", "mppi", "penalty_mppi", "mppi_cbf", "shield_mppi", "sc_mppi", "gs_mppi"]

# 1 main plot + 6 smaller plots
fig = plt.figure(figsize=(16, 8))

# Use GridSpec to define a 3x4 grid
# The large plot takes up the left 3 columns (all rows)
# The 6 smaller plots are stacked in the rightmost column, 2 per row
grid = fig.add_gridspec(3, 4)

axes = []
# Create main axis (BR-MPPI) spanning rows 0-2, cols 0-2
main_ax = fig.add_subplot(grid[:, :3])
axes.append(main_ax)

# Create small axes in the rightmost column
for i in range(6):
    row = i // 2
    # We want them in pairs but we only have 1 column left in our 3x4 grid. 
    # Let's adjust the gridspec to 3x5 instead.
plt.close(fig)
