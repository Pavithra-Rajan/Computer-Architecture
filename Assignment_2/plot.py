import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
# L1 v/s Hit rate
x_values = [16, 32, 64]
plt.ylim(88.08,91.04)
y1_values = [89.086, 89.540, 90.048]
y2_values = [89.082, 89.538, 90.047]
y3_values = [89.081, 89.538, 90.047]

plt.plot(x_values, y1_values, marker='o', label='2MB L2 cache')
# plt.plot(x_values, y2_values, marker='o', label='4MB L2 cache')
# plt.plot(x_values, y3_values, marker='o', label='8MB L2 cache')
plt.xticks(x_values)

plt.xlabel('L1 Cache size (KB)')
plt.ylabel('L1 Data hit rate')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.legend()

plt.savefig('FIGURES/l1_vs_l1hit.png')
plt.clf()

# L2 v/s Hit rate
x_values = [2, 4, 8]
# y1_values = [0.71, 0.76, 0.82]
# y2_values = [0.70, 0.74, 0.81]
# y3_values = [0.68, 0.73, 0.80]

y1_values = [71.43, 75.64, 82.13]
y2_values = [70.06, 74.45, 81.27]
y3_values = [68.50, 73.12, 80.28]

plt.plot(x_values, y1_values, marker='o', label='16KB L1 cache')
plt.plot(x_values, y2_values, marker='o', label='32KB L1 cache')
plt.plot(x_values, y3_values, marker='o', label='64KB L1 cache')
plt.xticks(x_values)

plt.xlabel('L2 Cache size (MB)')
plt.ylabel('L2 hit rate in %')
plt.legend()

plt.savefig('FIGURES/l2_vs_l2hit.png', bbox_inches='tight', pad_inches=0.2)
plt.clf()

# L2 v/s L2 Miss rate
x_values = [2, 4, 8]
y1_values = [28.56, 24.35, 17.86]
y2_values = [30, 25.54, 18.72]
y3_values = [31.49, 26.87, 19.71]

plt.plot(x_values, y1_values, marker='o', label='16KB L1 cache')
plt.plot(x_values, y2_values, marker='o', label='32KB L1 cache')
plt.plot(x_values, y3_values, marker='o', label='64KB L1 cache')
plt.xticks(x_values)

plt.xlabel('L2 Cache size (MB)')
plt.ylabel('L2 miss rate in %')
plt.legend()

plt.savefig('FIGURES/l2_vs_l2miss.png')
plt.clf()

# L2 v/s AMAT for L2 cache
x_values = [2, 4, 8]
y1_values = [38.57, 34.36, 27.86]
y2_values = [39.94, 35.54, 28.73]
y3_values = [41.49, 36.88, 29.72]

plt.plot(x_values, y1_values, marker='o', label='16KB L1 cache')
plt.plot(x_values, y2_values, marker='o', label='32KB L1 cache')
plt.plot(x_values, y3_values, marker='o', label='64KB L1 cache')
plt.xticks(x_values)

plt.xlabel('L2 Cache size (MB)')
plt.ylabel('AMAT for L2 in ns')
plt.legend()
y_interval = 1
y_locator = MultipleLocator(y_interval)
plt.gca().yaxis.set_major_locator(y_locator)

plt.savefig('FIGURES/l2_vs_AMAT_l2.png')
plt.clf()