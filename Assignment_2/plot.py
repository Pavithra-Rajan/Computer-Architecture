import matplotlib.pyplot as plt

# L1 v/s Hit rate
x_values = [16, 32, 64]
y_values = [0.891, 0.895, 0.900]

plt.plot(x_values, y_values, label='2MB L2 cache')
plt.xticks(x_values)

plt.xlabel('L1 Cache size (KB)')
plt.ylabel('L1 hit rate')
plt.legend()

plt.savefig('FIGURES/l1_vs_l1hit.png')
plt.clf()

# L2 v/s Hit rate
x_values = [2, 4, 8]
y1_values = [0.71, 0.76, 0.82]
y2_values = [0.70, 0.74, 0.81]
y3_values = [0.68, 0.73, 0.80]

plt.plot(x_values, y1_values, label='16KB L1 cache')
plt.plot(x_values, y2_values, label='32KB L1 cache')
plt.plot(x_values, y3_values, label='64KB L1 cache')
plt.xticks(x_values)

plt.xlabel('L2 Cache size (MB)')
plt.ylabel('L2 hit rate')
plt.legend()

plt.savefig('FIGURES/l2_vs_l2hit.png')
plt.clf()

# L2 v/s Hit rate
x_values = [2, 4, 8]
y1_values = [0.29, 0.24, 0.17]
y2_values = [0.30, 0.26, 0.19]
y3_values = [0.31, 0.27, 0.20]

plt.plot(x_values, y1_values, label='16KB L1 cache')
plt.plot(x_values, y2_values, label='32KB L1 cache')
plt.plot(x_values, y3_values, label='64KB L1 cache')
plt.xticks(x_values)

plt.xlabel('L2 Cache size (MB)')
plt.ylabel('L2 hit rate')
plt.legend()

plt.savefig('FIGURES/l2_vs_l2miss.png')
plt.clf()