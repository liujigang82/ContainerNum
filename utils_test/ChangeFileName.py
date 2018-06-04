import os

folder = "../img/"
i = 0

for filename in os.listdir("../img"):
    print("~~~~~~:", filename)
    i = i + 1
    print("%04d.jpg" %(i))
    os.rename(folder + filename, folder + "%04d.jpg" %(i))