# Sasano_cup
Detecting anomalies and classifying them into two classes: Flushes and Other.

# Installing requirements
## Windows :
pip install -r requirements.txt

# Dataset :
- ## images.zip: 
    Original images where we're going to detect the anomalies.
- ## dataset.zip: 
    Data splitted into train and test.

# Check if the directory structure is as follows.
- dataset
- images
- results
- weights
    - nn
    - composed

# Running the scripts:
## 1 / Prepare the dataset :
- ### dataset/flushes/train : training flushes images
- ### dataset/flushes/test : testing flushes images
- ### dataset/other/train : training other images
- ### dataset/other/test : test other images

## 2 /Data augmentation(FLUSH) :
create folder :
- ### dataset/flushes/train/augmented : 
    put the images you want to augment ! or all of them.

python data_augmentation.py

## 3 /Classification algorithm :
python classification_algorithm.py : Extracting the weights and saving them into the weights folder.

## 4 /Main process :


python main_process.py : Saving the new results in reults folder with detected anomalies from the images in Images folder.






