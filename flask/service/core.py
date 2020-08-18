import logging
import os
from flask import Flask, jsonify, request, send_file
app = Flask(__name__)

package_dir = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def hello():
    return "Hello World! <3,Flask"

@app.route('/health')
def health_check():
    return "OK"

@app.errorhandler(500)
def server_error(e):
    logging.exception("An error occurred during a request.")
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

@app.route('/train/<start_date>/<train_period>/<arguments>', methods=["POST"])
def main(start_date, train_period, arguments):
    """activate pytrain_esstimator to start training

    Args:
        start_date (str): start date of training
        train_period (int): duration (days) of training period
        arguments (str): Information for model created

    Returns:
        json: [description]
    """
    try:
        os.popen("cd '../src' && python main.py -s={0} -p={1} {2} &".
                 format(start_date, train_period, arguments))
        # os.popen("cd '../pytrain_estimator' && python main.py -s=" + start_date +
        #          " -t=" + str(train_period) +
        #          " " + str(arguments) + " &")
        return jsonify("Start new training process with setting start date as " + start_date +
                       ", training period as " + str(train_period) +
                       ", other arguments as " + arguments)
    except Exception as ex:
        logging.error(request.args)
        logging.error("Invalid argument parameters: %s", ex, exc_info=True)
        return 'Some error happened: {0}'.format(ex)


@app.route('/monitor/get_result_csv', methods=["GET"])
def get_result_csv():
    """Get the training list of models

    Returns:
        file: csv
    """
    try:
        file = "../../outputs/results.csv"
        return send_file(file)
    except Exception as ex:
        logging.error(request.args)
        return 'Some error happened: {0}'.format(ex)

if __name__ == '__main__':
    app.run(host="127.0.0.1", debug=True)
