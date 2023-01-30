from skimage import io

labelData = io.imread("Landsat_044033_20210711.tif")
print(labelData.shape)

imageData = io.imread("LC08_L1TP_044033_20210711_20210720_02_T1_B6.TIF")
print(imageData.shape)