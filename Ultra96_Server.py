import argparse
import base64
import os
import random
import socket
import sys
import threading
import time
import traceback

import numpy as np
from Crypto.Cipher import AES
from joblib import load

# Week 13 test: 8 moves, so 33 in total = (8*4) + 1 (logout)
# Week 9 and 11 tests: 3 moves, repeated 4 times each = 12 moves.
ACTIONS = ["gun", "sidepump", "hair"]
POSITIONS = ["1 2 3", "3 2 1", "2 3 1", "3 1 2", "1 3 2", "2 1 3"]
LOG_DIR = os.path.join(os.path.dirname(__file__), "evaluation_logs")
NUM_MOVE_PER_ACTION = 4
N_TRANSITIONS = 6
MESSAGE_SIZE = 4  # dancer_id, RTT, offset and raw_data

# The IP address of the Ultra96, testing part will be "127.0.0.1"
IP_ADDRESS = "127.0.0.1"
# The port number for three different dancer's laptops
PORT_NUM = [9091, 9092, 9093]
# Group ID number
GROUP_ID = 18


class Server(threading.Thread):
    def __init__(
        self,
        ip_addr,
        port_num,
        group_id,
        secret_key,
        n_moves=len(ACTIONS) * NUM_MOVE_PER_ACTION,
    ):
        super(Server, self).__init__()

        # Time stamps
        # Indicate the time when the server receive the package
        self.t2 = 0
        # Indicate the time when the server send the package
        self.t3 = 0

        # setup moves
        self.actions = ACTIONS
        self.position = POSITIONS
        self.n_moves = int(n_moves)

        # the moves should be a random distribution
        self.move_idxs = list(range(self.n_moves)) * NUM_MOVE_PER_ACTION
        assert self.n_moves == len(self.actions) * NUM_MOVE_PER_ACTION
        self.action = None
        self.action_set_time = None

        self.idx = 0
        self.timeout = 60
        self.has_no_response = False
        self.connection = None
        self.timer = None
        self.logout = False

        self.dancer_positions = ["1", "2", "3"]

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        print("starting up on %s port %s" % server_address)
        self.socket.bind(server_address)

        # Listen for incoming connections
        self.socket.listen(4)
        self.client_address, self.secret_key = self.setup_connection(secret_key)

        self.BUFFER = []

    def decrypt_message(self, cipher_text):
        # The data format which will be used here will be "raw data | t0 | RTT | offset | start_flag | muscle_fatigue"
        decoded_message = base64.b64decode(cipher_text)
        iv = decoded_message[:16]
        secret_key = bytes(str(self.secret_key), encoding="utf8")

        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        decrypted_message = cipher.decrypt(decoded_message[16:]).strip()
        decrypted_message = decrypted_message.decode("utf8")

        decrypted_message = decrypted_message[decrypted_message.find("#") :]
        decrypted_message = bytes(decrypted_message[1:], "utf8").decode("utf8")

        messages = decrypted_message.split("|")
        # print("messages decrypted:"+str(messages))

        dancer_id, RTT, offset, raw_data = messages[:MESSAGE_SIZE]
        return {
            "dancer_id": dancer_id,
            "RTT": RTT,
            "offset": offset,
            "raw_data": raw_data,
        }

    def inference(self):
        inputs = np.array(self.BUFFER)
        n_readings = 90
        start_time_step = 30
        num_time_steps = 60
        if inputs.shape[0] >= n_readings:
            # yaw pitch roll accx accy accz
            inputs = inputs[start_time_step : start_time_step + num_time_steps]
            inputs = np.array(
                [
                    [
                        inputs[:, 0],
                        inputs[:, 1],
                        inputs[:, 2],
                        inputs[:, 3],
                        inputs[:, 4],
                        inputs[:, 5],
                    ]
                ]
            )
            if debug:
                if model_type == "svc":
                    inputs = svc_utils.extract_raw_data_features(
                        inputs
                    )  # extract features
                    inputs = svc_utils.scale_data(inputs, scaler)  # scale features
                    predicted = model.predict(inputs)[0]
                    dance_move = ACTIONS[predicted]
                    print("Predicted:", dance_move)
                elif model_type == "dnn":
                    inputs = dnn_utils.extract_raw_data_features(
                        inputs
                    )  # extract features
                    inputs = dnn_utils.scale_data(inputs, scaler)  # scale features
                    inputs = torch.tensor(inputs)  # convert to tensor
                    outputs = model(inputs.float())
                    _, predicted = torch.max(outputs.data, 1)
                    dance_move = ACTIONS[predicted]
                    print("Predicted:", dance_move)
                else:
                    raise Exception("Model is not supported")
            if production:
                inputs = dnn_utils.extract_raw_data_features(inputs)  # extract features
                inputs = common_utils.scale_data(inputs, scaler)  # scale features
                inputs = [int(x * FIXED_FACTOR) for x in inputs]
                tc.write(inputs)
                tc.run()
                result = tc.get_result()
                predicted = np.argmax(result)
                dance_move = ACTIONS[predicted]
                print("Predicted:", dance_move)

            self.BUFFER = list()

    def run(self):
        while not self.shutdown.is_set():
            data = self.connection.recv(1024)
            self.t2 = time.time()
            if verbose:
                print("t2:" + str(self.t2))

            if data:
                try:
                    msg = data.decode("utf8")
                    decrypted_message = self.decrypt_message(msg)
                    if verbose:
                        print(
                            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                            + "messages received from dancer "
                            + str(decrypted_message["dancer_id"])
                        )
                        print(decrypted_message)
                    raw_data = decrypted_message["raw_data"]
                    raw_data = [float(x) for x in raw_data.split(" ")]

                    self.BUFFER.append(raw_data)
                    self.inference()
                    self.send_timestamp()

                except Exception:
                    print(traceback.format_exc())
            else:
                print("no more data from", self.client_address)
                self.stop()

    def send_timestamp(self):
        self.t3 = time.time()
        timestamp = str(self.t2) + "|" + str(self.t3)

        if verbose:
            print("t3:" + str(self.t3))
            print("timestamp: " + timestamp)

        self.connection.sendall(timestamp.encode())

    def setup_connection(self, secret_key):
        random.shuffle(self.move_idxs)
        print("No actions for 60 seconds to give time to connect")
        self.timer = threading.Timer(self.timeout, self.send_timestamp)
        self.timer.start()

        # Wait for a connection
        print("waiting for a connection")
        self.connection, client_address = self.socket.accept()

        print("Enter the secret key: ")
        if not secret_key:
            secret_key = sys.stdin.readline().strip()

        print("connection from", client_address)
        if len(secret_key) == 16 or len(secret_key) == 24 or len(secret_key) == 32:
            pass
        else:
            print("AES key must be either 16, 24, or 32 bytes long")
            self.stop()

        return client_address, secret_key  # forgot to return the secret key

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()


def main(dancer_id, secret_key):
    if dancer_id == 0:
        dancer_server0 = Server(IP_ADDRESS, PORT_NUM[0], GROUP_ID, secret_key)
    if dancer_id == 1:
        dancer_server1 = Server(IP_ADDRESS, PORT_NUM[1], GROUP_ID, secret_key)
    if dancer_id == 2:
        dancer_server2 = Server(IP_ADDRESS, PORT_NUM[2], GROUP_ID, secret_key)

    if dancer_id == 0:
        dancer_server0.start()
        print(
            "dancer_server0 started: IP address:"
            + IP_ADDRESS
            + " Port Number: "
            + str(PORT_NUM[0])
            + " Group ID number: "
            + str(GROUP_ID)
        )
    if dancer_id == 1:
        dancer_server1.start()
        print(
            "dancer_server1 started: IP address:"
            + IP_ADDRESS
            + " Port Number: "
            + str(PORT_NUM[1])
            + " Group ID number: "
            + str(GROUP_ID)
        )
    if dancer_id == 2:
        dancer_server2.start()
        print(
            "dancer_server1 started: IP address:"
            + IP_ADDRESS
            + " Port Number: "
            + str(PORT_NUM[2])
            + " Group ID number: "
            + str(GROUP_ID)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External Comms")
    parser.add_argument("--dancer_id", help="dancer id", type=int, required=True)
    parser.add_argument("--debug", default=False, help="debug mode", type=bool)
    parser.add_argument(
        "--production", default=False, help="production mode", type=bool
    )
    parser.add_argument("--verbose", default=False, help="verbose", type=bool)
    parser.add_argument("--model_type", help="svc or dnn or ultra96 model")
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--scaler_path", help="path to scaler")
    parser.add_argument("--secret_key", default="1234123412341234", help="secret key")
    parser.add_argument(
        "--bit_path",
        default="/home/xilinx/jupyter_notebooks/frontier/capstone_full.bit",
        help="path to bit",
    )

    args = parser.parse_args()
    dancer_id = args.dancer_id
    debug = args.debug
    production = args.production
    verbose = args.verbose
    model_type = args.model_type
    model_path = args.model_path
    scaler_path = args.scaler_path
    secret_key = args.secret_key
    bit_path = args.bit_path

    print("dancer_id:", dancer_id)
    print("debug:", debug)
    print("production:", production)
    print("verbose:", verbose)
    print("model_type:", model_type)
    print("model_path:", model_path)
    print("scaler_path:", scaler_path)
    print("secret_key:", secret_key)

    if debug:
        scaler = load(scaler_path)
        if model_type == "svc":
            import svc_utils

            model = load(model_path)
        elif model_type == "dnn":
            import torch

            import dnn_utils

            model = dnn_utils.DNN()
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            raise Exception("Model is not supported")
    if production:
        import common_utils
        from technoedge import FIXED_FACTOR, TechnoEdge

        scaler = load(scaler_path)
        tc = TechnoEdge(bit_path)
        f = open(model_path, "rb")
        wts = np.load(f)
        tc.put_weights(wts)
        f.close()

    main(dancer_id, secret_key)
