# !/usr/bin/env python3
import http.server as SimpleHTTPServer
import socketserver as SocketServer
import pandas as pd
import time


class GetHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self):
        start_time = time.time()
        self._set_headers()
        print("Get Request Received...")
        print("Command: ", self.command)
        print("Path: ", self.path)
        content = get_next_data(data=data)
        self.wfile.write(content.encode("utf8"))
        print("--- %s seconds to get the data---" % (time.time() - start_time))


def get_next_data(data):
    global start_index
    start, end = get_start_end_index()
    start_index += DATA_OFFSET
    result = data.iloc[start:end]
    mean_sum = result.mean()
    print("Actual Feedback: ", mean_sum['Feedback'])
    result = result.drop(columns=['Feedback'])
    return result.to_json()


def get_start_end_index():
    start = start_index
    end = start_index + DATA_OFFSET
    # Check the outer bound
    if end > number_of_rows:
        end = number_of_rows
    if start > number_of_rows:
        start = number_of_rows

    return start, end


individual_number = 19
file_name = "individual_data-" + str(individual_number) + ".csv"
if __name__ == '__main__':
    PORT = 8800
    DATA_OFFSET = 61  # In order to look back 60 data we need 61 examples
    data = pd.read_csv("data/test_data.csv", delimiter=',', index_col=0)  # TODO: REPLACE WITH TEST FILE
    data = data[data["Individual"] == individual_number]
    data.to_csv(file_name)
    data = data.drop(columns=['Individual', 'Epoch'])
    start_index = 0
    number_of_rows = len(data)
    # Run Server
    Handler = GetHandler
    httpd = SocketServer.TCPServer(("", PORT), Handler)
    print("Data Simulator Started...")
    httpd.serve_forever()
