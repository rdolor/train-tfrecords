# Customized training using TFRecords data format

This is a project using TF to train and test some TFRecords data. The models and training pipeline for click-through rate prediction are customized.

## Dataset
The  data comes from the text files of the  standardized data format of [iPinYou RTB dataset](https://github.com/wnzhang/make-ipinyou-data). Using [this project](https://github.com/rdolor/data-to-tfrecords), text files are transformed into TFRecords file.

### Data information:
```
feature = click,          shape = (10000, 1),  Unique count = 2,   min = 0,   max = 1
feature = weekday,        shape = (10000, 1),  Unique count = 1,   min = 4,   max = 4
feature = region,         shape = (10000, 1),  Unique count = 35,  min = 0,   max = 395
feature = city,           shape = (10000, 1),  Unique count = 359, min = 0,   max = 399
feature = adexchange,     shape = (10000, 1),  Unique count = 3,   min = 1,   max = 3
feature = slotformat,     shape = (10000, 1),  Unique count = 2,   min = 0,   max = 1
feature = hour,           shape = (10000, 1),  Unique count = 1,   min = 0,   max = 0
feature = slotwidth,      shape = (10000, 1),  Unique count = 6,   min = 160, max = 1000
feature = slotheight,     shape = (10000, 1),  Unique count = 4,   min = 90,  max = 600
feature = slotvisibility, shape = (10000, 1),  Unique count = 4,   min = 0,   max = 255
feature = slotprice,      shape = (10000, 1),  Unique count = 46,  min = 0,   max = 280
feature = usertag,        shape = (10000, 39), Unique count = 45,  min = -1,  max = 16706
```

## How to create the environment

**1. Using pipenv**

* To create or activate a virtual env: `pipenv shell`

    * Install all required packages:
        * install packages exactly as specified in **Pipfile.lock**: `pipenv sync`
        * install using the **Pipfile**, including the dev packages: `pipenv install --dev`

**2. Using docker**

* Build the image: `make build`
* Create a container: `docker run -it --rm train_tfrecords:master bash`

## How to run the program
* Testing the code: `make tests`
* Training and Testing on data:
    - For easy configurations, edit: `src/initial_configurations/default`
    - Run training: `make train`
* Tuning the model:
    - Go inside the folder `cd tune/`.
    - Edit the configurations in `search_space.json` and `tune_config.yml`.
    - Run the tuner: `nnictl create --config tune_config.yml --port <PORT_NUMBER>`






