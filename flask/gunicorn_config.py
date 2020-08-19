"""
config for Gunicorn service
"""

import os
PORT = os.environ["PORT"] if "PORT" in os.environ else "7777"

bind = "0.0.0.0:" + PORT
workers = 1
accesslog = "-"
errorlog = "-"
capture_output = True
loglevel = "info"