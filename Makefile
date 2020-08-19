IMAGE_NAME=train_tfrecords:master

build ::
	echo ${IMAGE_NAME}
	docker build -f ./docker/Dockerfile -t ${IMAGE_NAME} .
	#docker push ${IMAGE_NAME}

tests ::
	pytest tests

train ::
	python -m src.main

activate_flask ::
	cd /flask/ && ./activate_flask 
