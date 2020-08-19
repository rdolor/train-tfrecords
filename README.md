# Customized training using TFRecords data format

This project uses TensorFlow and some TFRecords data for training. The models and training pipeline for click-through rate prediction are customized. 

Flask is used to create a service that can (re)train a model. 

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
*   To build the image: `make build`
*   To create the container: `docker-compose up -d`
    - Note that this will already trigger the Flask service app.
    - To get the exact IP address, try running `docker-machine inspect default | grep IPAddress` and then use this address instead of `localhost`.
*   To get into the container: `docker exec -it <container_name> bash`
*   To check the logs, especially when training the model: `docker logs -f <container_name>`
*   To force stop the container: `docker rm -f <container_name>`


## How to run the program

* Testing the code: `make tests`
* Training and Testing on data:
    - For easy configurations, edit: `src/initial_configurations/default`
    - Run training: `make train`
* Tuning the model:
    - Go inside the folder `cd tune/`.
    - Edit the configurations in `search_space.json` and `tune_config.yml`.
    - Run the tuner: `nnictl create --config tune_config.yml --port <PORT_NUMBER>`
* How to use the Flask service:
    - To check if it is working, GET `http://localhost:7777/`, should result to:
    ```
    Hello World! <3,Flask
    ```
    - To (re)train a model, POST on `http://localhost:7777/train/<start_date>/<training_period>`.
    - To check the performance metrics of the trained models, GET `http://localhost:7777/monitor/get_result_csv`.