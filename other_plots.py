import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# STGCN_days = [231.38,
#     220.72,
#     232.01,
#     234.92,
#     239.78,
#     230.39,
#     219.14,
#     218.53,
#     218.83]

# SVR_days = [260.84,
#     223.10,
#     233.19,
#     237.19,
#     243.08,
#     228.08,
#     214.84,
#     225.80,
#     223.21]

# days = ["Monday (1)", "Tuesday (1)", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday (2)", "Tuesday (2)"]

# plt.xlabel("Test Days")
# plt.ylabel("MAE (vehicles/hour)")
# plt.plot(days, STGCN_days, label="STGCN")
# plt.plot(SVR_days, label="SVR")
# plt.show()

orig = [162.74887,
    218.8375,
    214.01633,
    169.02692,
    91.5046,
    288.50305,
    342.279,
    192.71594,
    303.34744,
    249.49396,
    267.82867,
    227.3002]

ny = [151.75197,
    187.29227,
    230.12427,
    146.66322,
    87.88946,
    280.11792,
    342.50693,
    193.4136,
    318.69287,
    253.64438,
    272.43457,
    224.04832]

jf = [153.94156,
    215.5167,
    220.60606,
    142.7948,
    92.94032,
    276.9002,
    346.39807,
    185.67079,
    296.75864,
    227.90303,
    269.36923,
    220.79994]

mt = [157.87354,
    268.72437,
    229.30031,
    133.81488,
    78.23666,
    273.66306,
    329.03677,
    178.28165,
    294.4525,
    231.04187,
    262.5744,
    221.54546]

sh = [160.48222,
    134.9234,
    221.72324,
    171.52016,
    84.73047,
    271.32385,
    341.05807,
    187.12263,
    300.25577,
    236.03159,
    262.24002,
    215.58286]

fig, ax = plt.subplots()
ax.plot(orig, label="Original")
ax.plot(ny, label="NY")
ax.plot(jf, label="JF")
ax.plot(mt, label="MT")
ax.plot(sh, label="SH")

# ax.xaxis.set_major_locator(MaxNLocator(11)) 

labels = [item.get_text() for item in ax.get_xticklabels()]
print(labels)
# labels[1] = 'Testing'
ax.xaxis.set_ticks([0,1,2,3,4,5,6,7,8,9,10, 11])
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', "Average"]
ax.set_xticklabels(labels)
plt.title("MAE at Bravo Nodes")
plt.xlabel("Node")
plt.ylabel("MAE (vehicles/hour)")
plt.legend()
plt.show()
