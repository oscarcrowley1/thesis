import matplotlib.pyplot as plt

STGCN_days = [231.38,
    220.72,
    232.01,
    234.92,
    239.78,
    230.39,
    219.14,
    218.53,
    218.83]

SVR_days = [260.84,
    223.10,
    233.19,
    237.19,
    243.08,
    228.08,
    214.84,
    225.80,
    223.21]

days = ["Monday (1)", "Tuesday (1)", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday (2)", "Tuesday (2)"]

plt.xlabel("Test Days")
plt.ylabel("MAE (vehicles/hour)")
plt.plot(days, STGCN_days, label="STGCN")
plt.plot(SVR_days, label="SVR")
plt.show()
