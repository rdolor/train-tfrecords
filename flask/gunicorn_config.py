import os
APP_PREFIX = os.environ["APP_PREFIX"] if "APP_PREFIX" in os.environ else "False"
PORT = os.environ["PORT"] if "PORT" in os.environ else 7777

bind = "0.0.0.0:" + PORT
workers = 1
accesslog = "-"
errorlog = "-"
capture_output = True
loglevel = "info"