from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt

object = pd.read_pickle(r"checkpoints/losses.pk")

print(len(object))
count = 0
for thing in object:
    if count==0 or count == 1:
        print("XXXX")
        print(thing)
        print(len(thing))
        if count == 0:
            label_string = "Training Loss"
        else:
            label_string = "Validation Loss"
            
        plt.plot(thing, label=label_string)
            
    count = count + 1

plt.xlabel("Iterations")
plt.ylabel("Loss")
# plt.yscale("log")
plt.legend()
plt.show()

# plt.plot(object)
# plt.show()