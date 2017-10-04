import cv2

with open('TRAIN/1/0.bmp', 'rb') as imageFile:
    f = imageFile.read()
    b = bytearray(f)

print(b)