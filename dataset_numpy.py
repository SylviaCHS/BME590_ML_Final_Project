from os import listdir



filename = listdir('./TB_Image')
print(filename)
img = Image.open(infilename)
img.load()
data = np.asarray(img, dtype="int32")