import math
import sys
from pathlib import Path
import requests, zipfile, io, os

from GANDLF.utils import writeTrainingCSV
from GANDLF.parseConfig import parseConfig

all_models_segmentation = ['unet', 'resunet', 'fcn', 'uinc'] # pre-defined segmentation model types for testing
all_models_regression = [] # populate once it becomes available

'''
steps to follow to write tests:
[x] download sample data
[x] construct the training csv
[ ] for each dir (application type) and sub-dir (image dimension), run training for a single epoch on cpu
[ ] for each dir (application type) and sub-dir (image dimension), run inference for a single trained model per testing/validation split for a single subject on cpu
4. hopefully the various sys.exit messages throughout the code will catch issues
'''

def test_download_data():
  '''
  This function downloads the sample data, which is the first step towards getting everything ready
  '''
  urlToDownload = 'https://github.com/sarthakpati/tempDownloads/raw/main/data.zip'
  if not Path(os.getcwd() + '/testing/data/test/3d_rad_segmentation/001/image.nii.gz').exists():
    print('Downloading and extracting sample data')
    r = requests.get(urlToDownload)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall('./testing')

def test_constructTrainingCSV():
  '''
  This function constructs training csv
  '''
  inputDir = os.path.normpath('./testing/data')
  # delete previous csv files
  files = os.listdir(inputDir)
  for item in files:
    if item.endswith(".csv"):
      os.remove(os.path.join(inputDir, item))

  for application_data in os.listdir(inputDir):
    currentApplicationDir = os.path.join(inputDir, application_data)

    if '2d_rad_segmentation' in application_data:
      channelsID = '_blue.png,_red.png,_green.png'
      labelID = 'mask.png'
    elif '3d_rad_segmentation' in application_data:
      channelsID = 'image'
      labelID = 'mask'
    writeTrainingCSV(currentApplicationDir, channelsID, labelID, inputDir + '/train_' + application_data + '.csv')


def test_segmentation_rad_2d():
  print('Starting 2D Rad segmentation tests')

  inputDir = os.path.normpath('./testing/data')
  parameters = parseConfig(inputDir + '/2d_rad_segmentation/sample_training.yaml')
  for model in all_models_segmentation:
    parameters['model']['architecture'] = model 

  print('passed')

def test_segmentation_rad_3d():
  print('Starting 3D Rad segmentation tests')
  
  inputDir = os.path.normpath('./testing/data')
  parameters = parseConfig(inputDir + '/3d_rad_segmentation/sample_training.yaml')
  for model in all_models_segmentation:
    parameters['model']['architecture'] = model 

  print('passed')
