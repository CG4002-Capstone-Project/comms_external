import argparse
import base64
import random
import socket
import sys
import threading
import time
import traceback
import warnings

import numpy as np
import pika
from Crypto import Random
from Crypto.Cipher import AES
from joblib import load

warnings.filterwarnings("ignore")
# Week 13 test: 8 moves, so 33 in total = (8*4) + 1 (logout)
# Week 9 and 11 tests: 3 moves, repeated 4 times each = 12 moves.
ACTIONS = ["gun", "sidepump", "hair"]
POSITIONS = ["1 2 3", "3 2 1", "2 3 1", "3 1 2", "1 3 2", "2 1 3"]
NUM_MOVE_PER_ACTION = 4
N_TRANSITIONS = 6
MESSAGE_SIZE = 4  # dancer_id, RTT, offset and raw_data
ENCRYPT_BLOCK_SIZE = 16

# The IP address of the Ultra96, testing part will be "127.0.0.1"
IP_ADDRESS = "127.0.0.1"
# The port number for three different dancer's laptops
PORT_NUM = [9091, 9092, 9093]
# Group ID number
GROUP_ID = 18


class Client(threading.Thread):
    def __init__(self, ip_addr, port_num, group_id, key):
        super(Client, self).__init__()

        self.idx = 0
        self.timeout = 60
        self.has_no_response = False
        self.connection = None
        self.timer = None
        self.logout = False

        self.group_id = group_id
        self.key = key

        self.dancer_positions = ["1", "2", "3"]

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        print(
            "Start connecting... server address: %s port: %s" % server_address,
            file=sys.stderr,
        )
        self.socket.connect(server_address)
        print("Connected")

    # To encrypt the message, which is a string
    def encrypt_message(self, message):
        raw_message = "#" + message
        print("raw_message: " + raw_message)
        padded_raw_message = raw_message + " " * (
            ENCRYPT_BLOCK_SIZE - (len(raw_message) % ENCRYPT_BLOCK_SIZE)
        )
        print("padded_raw_message: " + padded_raw_message)
        iv = Random.new().read(AES.block_size)
        secret_key = bytes(str(self.key), encoding="utf8")
        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        encrypted_message = base64.b64encode(
            iv + cipher.encrypt(bytes(padded_raw_message, "utf8"))
        )
        print("encrypted_message: ", encrypted_message)
        return encrypted_message

    # To send the message to the sever
    def send_message(self, message):
        encrypted_message = self.encrypt_message(message)
        print("Sending message:", encrypted_message)
        self.socket.sendall(encrypted_message)

    def receive_dancer_position(self):
        dancer_position = self.socket.recv(1024)
        msg = dancer_position.decode("utf8")
        return msg

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()


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
        self.idle_state = True
        self.idle_index = 0

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
                    if eval_server:
                        my_client.send_message(
                            "1 2 3" + "|" + dance_move + "|" + "1.5" + "|"
                        )
                    if dashboard:
                        channel.basic_publish(
                            exchange="", routing_key="results", body=dance_move
                        )
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
                    if eval_server:
                        my_client.send_message(
                            "1 2 3" + "|" + dance_move + "|" + "1.5" + "|"
                        )
                    if dashboard:
                        channel.basic_publish(
                            exchange="", routing_key="results", body=dance_move
                        )
                else:
                    raise Exception("Model is not supported")
            if production:
                inputs = common_utils.extract_raw_data_features(
                    inputs
                )  # extract features
                inputs = common_utils.scale_data(inputs, scaler)  # scale features
                inputs = list(inputs[0])
                inputs = [int(x * FIXED_FACTOR) for x in inputs]
                tc.write(inputs)
                tc.run()
                result = tc.get_result()
                predicted = np.argmax(result)
                dance_move = ACTIONS[predicted]
                cpu_usage, fpga_usage = get_power()
                print(
                    "CPU Usage: ",
                    cpu_usage,
                    "FPGA Usage: ",
                    fpga_usage,
                    "Predicted:",
                    dance_move,
                )
                if eval_server:
                    my_client.send_message(
                        "1 2 3" + "|" + dance_move + "|" + "1.5" + "|"
                    )
                if dashboard:
                    channel.basic_publish(
                        exchange="", routing_key="results", body=dance_move
                    )

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

                    mode = int(raw_data[0])
                    if self.idle_state:
                        if mode == 1:
                            self.idle_index += 1
                            if self.idle_index % 30 == 0:
                                print("Idling")
                        else:
                            self.idle_state = False
                    else:
                        # Minus the idle state
                        self.BUFFER.append(raw_data[1:])
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
    parser.add_argument(
        "--eval_server", default=False, help="send to eval server", type=bool
    )
    parser.add_argument("--ip_addr", default="localhost", help="eval server ip")
    parser.add_argument("--port_num", default="8000", help="eval server port", type=int)
    parser.add_argument("--group_id", default="18", help="group number")
    parser.add_argument(
        "--key", default="1234123412341234", help="secret key", type=int
    )
    parser.add_argument(
        "--dashboard", default=False, help="send to dashboard", type=bool
    )
    parser.add_argument(
        "--cloudamqp_url",
        default="amqps://yjxagmuu:9i_-oo9VNSh5w4DtBxOlB6KLLOMLWlgj@mustang.rmq.cloudamqp.com/yjxagmuu",
        help="dashboard connection",
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
    eval_server = args.eval_server
    ip_addr = args.ip_addr
    port_num = args.port_num
    group_id = args.group_id
    key = args.key
    dashboard = args.dashboard
    cloudamqp_url = args.cloudamqp_url

    print("dancer_id:", dancer_id)
    print("debug:", debug)
    print("production:", production)
    print("verbose:", verbose)
    print("model_type:", model_type)
    print("model_path:", model_path)
    print("scaler_path:", scaler_path)
    print("secret_key:", secret_key)
    print("eval_server:", eval_server)
    print("bit_path:", bit_path)
    print("ip_addr:", ip_addr)
    print("port_num:", port_num)
    print("group_id:", group_id)
    print("key:", key)
    print("dashboard:", dashboard)
    print("cloudamqp_url:", cloudamqp_url)

    if eval_server:
        my_client = Client(ip_addr, port_num, group_id, key)

    if dashboard:
        params = pika.URLParameters(cloudamqp_url)
        params.socket_timeout = 5
        connection = pika.BlockingConnection(params)  # Connect to CloudAMQP
        channel = connection.channel()  # start a channel
        channel.queue_declare(queue="results")  # Declare a queue

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
        from technoedge import FIXED_FACTOR, TechnoEdge, get_power

        scaler = load(scaler_path)
        tc = TechnoEdge(bit_path)
        f = open(model_path, "rb")
        wts = np.load(f)
        tc.put_weights(wts)
        f.close()

    main(dancer_id, secret_key)
