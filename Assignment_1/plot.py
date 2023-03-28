import matplotlib.pyplot as plt

with open('output.txt', 'r') as f:
    lines = f.readlines()

data = []

for line in lines:
    values = line.strip().split()
    data.append(tuple(values))

# print(data[0:10])
# print(type(data[0]))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
          '#bcbd22', '#17becf', '#ff00ff', '#000000',
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
# print(int_sizes)
sorted_result = {str(i): result_dict[str(i)] for i in int_sizes}
# print(sorted_result)

# print(list(result_dict.keys()))
legends = list(sorted_result.keys())
# print(result_dict[legends[0]][0])
# print(list(zip(*result_dict[legends[0]]))[0])
plt.figure(figsize=(13,6))
plt.grid(axis="x", linestyle = '--')
for legend in legends:
    # print(legend)
    x = list(zip(*sorted_result[legend]))[0]
    y = list(zip(*sorted_result[legend]))[1]
    # print(type(x))
    # print(type(y))
    # print(x)
    # print(y)
    converted = []
    for i in y:
        converted.append(float(i))
    # print(converted)
    # print(max(converted))
    plt.plot(list(x), list(converted), label = legend) 

# print(len(legends))
plt.xlabel('stride')
plt.ylabel('read(ns)')
plt.tight_layout()
plt.savefig('plot.png')
plt.legend(loc='best')
plt.show()
