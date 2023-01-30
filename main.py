from PIL import Image
from skimage import io

print(io.imread("Landsat_044033_20180719.tif"))

# labelData = Image.open("Landsat_044033_20180719.tif").convert("RGB")
# image = Image.open("LC08_L1TP_044033_20180719_20200831_02_T1_B6.TIF").convert("RGB")

# print(test)