import http.server as SimpleHTTPServer
import logging
import socketserver as SocketServer
import pandas as pd
import requests
import cyber_sense_brain as predictor
import time

# SERVER CONFIG
GET_DATA_REQ = "GetPrediction"
SIMULATED_DATA_URL = "http://localhost:8800/NeuLogAPI"  # TODO: Change the start and end
PORT = 5555

# Set up log info
logging.basicConfig(filename='server.log', filemode='a',
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


def get_simulated_data():
    response = requests.get(url=SIMULATED_DATA_URL)
    if response.ok:
        json_data = response.json()
        print("Data Received...")
        data = prepare_data_frame(json_data)
        return data

    else:
        print("Cannot get Data")
        return False


def prepare_data_frame(json_data):
    data = {"HR": json_data["HR"], "PC_HR": json_data["PC_HR"], "HR_MIN": json_data["HR_MIN"],
            "HR_MAX": json_data["HR_MAX"],
            "HRV": json_data["HRV"], "PC_HRV": json_data["PC_HRV"],
            "HRV_MIN": json_data["HRV_MIN"], "HRV_MAX": json_data["HRV_MAX"], "PC_BR": json_data["PC_BR"],
            "GSR": json_data["GSR"], "PC_GSR": json_data["PC_GSR"], "GSR_MIN": json_data["GSR_MIN"],
            "GSR_MAX": json_data["GSR_MAX"]}

    return data


def request_data_from_sensors():
    logging.info("Requesting Data from Physiological Sensors...")
    minute_wise_data = get_simulated_data()
    logging.info("Successfully got cross_validation_data from sensors.")
    return pd.DataFrame(minute_wise_data)


class GetHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    # Waiting for http://localhost:5555/GetPrediction
    def do_GET(self):
        start_time = time.time()
        self._set_headers()
        print("Get Request Received...")
        print("Command: ", self.command)
        print("Path: ", self.path)
        if GET_DATA_REQ in self.path:
            data = request_data_from_sensors()
            print(data)
            cyber_sickness_prediction = predictor.predict_cybersickness(data, loaded_model)
            self.wfile.write(cyber_sickness_prediction.encode("utf8"))

        print("--- %s seconds to return cybersickness prediction---" % (time.time() - start_time))


def start_cyber_sense():
    Handler = GetHandler
    httpd = SocketServer.TCPServer(("", PORT), Handler)
    print("Starting CyberSense Server...")
    print("Waiting for CyberSense Client...")
    httpd.serve_forever()


# Run Experiment
if __name__ == '__main__':
    loaded_model = predictor.load_model_from_file()
    start_cyber_sense()
