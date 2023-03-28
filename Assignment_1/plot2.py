import matplotlib.pyplot as plt
import numpy as np

with open('output3.txt', 'r') as f:
    lines = f.readlines()

data = []

for line in lines:
    values = line.strip().split()
    converted_values = [int(values[0]), int(values[1]), float(values[2])]
    print(converted_values)
    data.append(tuple(values))

# print(data[0:10])
# print(type(data[0]))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
          '#bcbd22', '#17becf', '#ff00ff', '#000000'
         ]

result_dict = {}
for key in set(x[0] for x in data):
    # print(int(key))
    result_dict[key] = [(y[1], y[2]) for y in data if y[0] == key]

# print(result_dict)
sizes = list(result_dict.keys())
int_sizes = []
for k in sizes:
    int_sizes.append(int(k))

int_sizes.sort()
print(int_sizes)
sorted_result = {str(i): result_dict[str(i)] for i in int_sizes}
print(sorted_result)

# print(list(result_dict.keys()))
legends = list(sorted_result.keys())
# print(result_dict[legends[0]][0])
# print(list(zip(*result_dict[legends[0]]))[0])
plt.figure(figsize=(13,6))
plt.grid(axis="x", linestyle = '--')
for legend in legends:
    print(legend)
    x = list(zip(*sorted_result[legend]))[0]
    y = list(zip(*sorted_result[legend]))[1]
    # print(type(x))
    # print(type(y))
    print(x)
    # print(y)
    converted = []
    for i in y:
        converted.append(float(i))
    #print(converted)
    print(max(converted))
    x_log = [np.log2(int(i)) for i in x]
    y_log = [np.log2(i) for i in converted]
    plt.plot(x_log, y_log, label = legend) 

print(len(legends))
plt.xlabel('log of stride')
plt.ylabel('log of read(ns)')
plt.tight_layout()
plt.savefig('plot3.png')
plt.legend(loc='best')
plt.show()
