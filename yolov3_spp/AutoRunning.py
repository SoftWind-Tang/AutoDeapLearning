import os

for i in range(3):
    os.system("python calculate_dataset.py")
    os.system("python train.py")
    os.system("python predict_test.py")