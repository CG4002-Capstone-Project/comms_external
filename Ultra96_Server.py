#This Server is meant to receive the data sent from dancers laptops and delivers the required data for the machine learning part

import os
import sys
import random
import time

import socket
import threading

import base64
# import numpy as np
# from tkinter import Label, Tk
# import pandas as pd
from Crypto.Cipher import AES
from joblib import load

import dnn_utils
import svc_utils
import numpy as np
import torch

scaler_path = "./dnn_std_scaler.bin"
model_path = "./dnn_model.pth"
activities = ["dab", "gun", "elbow"]
scaler = load(scaler_path)

model = dnn_utils.DNN()
model.load_state_dict(torch.load(model_path))
model.eval()

# Week 13 test: 8 moves, so 33 in total = (8*4) + 1 (logout)
#ACTIONS = ['zigzag', 'rocket', 'hair', 'pushback', 'windowwipe', 'elbowlock', 'scarecrow', 'shouldershrug']
# Week 9 and 10 tests: 3 moves, repeated 4 times each = 12 moves.
ACTIONS = ['zigzag', 'rocket', 'hair']
POSITIONS = ['1 2 3', '3 2 1', '2 3 1', '3 1 2', '1 3 2', '2 1 3']
LOG_DIR = os.path.join(os.path.dirname(__file__), 'evaluation_logs')
NUM_MOVE_PER_ACTION = 4
N_TRANSITIONS = 6
MESSAGE_SIZE = 4 # dancer_id, RTT, offset and raw_data

# The IP address of the Ultra96, testing part will be "127.0.0.1"
IP_ADDRESS = "127.0.0.1"
# The port number for three different dancer's laptops
PORT_NUM = [9091, 9092, 9093]
# Group ID number
GROUP_ID = 18

#The buffer to store raw data


class Server(threading.Thread):
    def __init__(self, ip_addr, port_num, group_id, n_moves=len(ACTIONS) * NUM_MOVE_PER_ACTION):
        super(Server, self).__init__()

        # # setup logger
        # self.log_filename = 'group{}_logs.csv'.format(group_id)
        # if not os.path.exists(LOG_DIR):
        #     os.makedirs(LOG_DIR)
        # self.log_filepath = os.path.join(LOG_DIR, self.log_filename)
        # self.columns = ['timestamp', 'position', 'gt_position', 'action', 'gt_action', 'response_time', 'sync', 'is_action_correct', 'is_position_correct']
        # self.df = pd.DataFrame(columns=self.columns)
        # self.df = self.df.set_index('timestamp')

        #Time stamps
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

        self.dancer_positions = ['1', '2', '3']

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        print('starting up on %s port %s' % server_address)
        self.socket.bind(server_address)

        # Listen for incoming connections
        self.socket.listen(4)
        self.client_address, self.secret_key = self.setup_connection() 

        self.BUFFER = []
        self.dance_move = None


        

    def decrypt_message(self, cipher_text):
        #The data format which will be used here will be "raw data | t0 | RTT | offset | start_flag | muscle_fatigue"
        decoded_message = base64.b64decode(cipher_text)
        iv = decoded_message[:16]
        secret_key = bytes(str(self.secret_key), encoding="utf8") 

        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        decrypted_message = cipher.decrypt(decoded_message[16:]).strip()
        decrypted_message = decrypted_message.decode('utf8')

        decrypted_message = decrypted_message[decrypted_message.find('#'):]
        decrypted_message = bytes(decrypted_message[1:], 'utf8').decode('utf8')

        messages = decrypted_message.split('|')
        # print("messages decrypted:"+str(messages))

        dancer_id, RTT, offset, raw_data = messages[:MESSAGE_SIZE]
        return {
            'dancer_id':dancer_id,'RTT': RTT, 'offset':offset, "raw_data":raw_data
        }

    def inference(self):
        inputs = np.array(self.BUFFER)
        print(inputs.shape)
        print("Predicted dance move:", self.dance_move)
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
            # if model_type == "svc":
            #     inputs = svc_utils.extract_raw_data_features(
            #         inputs
            #     )  # extract features
            #     inputs = svc_utils.scale_data(inputs, scaler)  # scale features
            #     predicted = model.predict(inputs)[0]
            #     dance_move = activities[predicted]
            # elif model_type == "dnn":
            inputs = dnn_utils.extract_raw_data_features(
                inputs
            )  # extract features
            inputs = dnn_utils.scale_data(inputs, scaler)  # scale features
            inputs = torch.tensor(inputs)  # convert to tensor
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            self.dance_move = activities[predicted]
            
            # else:
            #     raise Exception("Model is not supported")
            self.BUFFER = list()


    def run(self):
        while not self.shutdown.is_set():
            data = self.connection.recv(1024)
            self.t2 = time.time()
            print("t2:" + str(self.t2))

            if data:
                try:
                    msg = data.decode("utf8")
                    decrypted_message = self.decrypt_message(msg)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + "messages received from dancer " + str(decrypted_message["dancer_id"]))
                    print(decrypted_message)
                    raw_data = decrypted_message['raw_data']
                    raw_data = [float(x) for x in raw_data.split(" ")]

                    self.BUFFER.append(raw_data)
                    # print("BUFFER:", BUFFER)
                    self.inference()
                    # if decrypted_message['action'] == "logout":
                    #     self.logout = True
                    #     self.stop()
                    #     print("bye bye")
                    # elif len(decrypted_message['action']) == 0:
                    #     pass  # no valid action sent
                    # elif self.action is None:
                    #     pass  # no action sent yet.
                    # else:  # action is available so we log it
                    #     self.has_no_response = False
                    #     self.write_move_to_logger(decrypted_message['position'], decrypted_message['action'], decrypted_message['sync'])

                    #     print("{} :: {} :: {}".format(decrypted_message['position'],
                    #                                               decrypted_message['action'], 
                    #                                               decrypted_message['sync']))
                    self.send_timestamp()
                        # self.send_dancer_positions()
                        # self.set_next_action()  # Get new action
                except Exception as e:
                    print(e)
            else:
                print('no more data from', self.client_address)
                self.stop()


    def send_timestamp(self):
        self.t3 = time.time()
        print("t3:" + str(self.t3))

        timestamp = str(self.t2) + "|" + str(self.t3)
        print("timestamp: " + timestamp)
        self.connection.sendall(timestamp.encode())
        


    # def send_dancer_positions(self):
    #     dancer_positions = self.dancer_positions
    #     print('New Dancer Positions: {0}'.format(dancer_positions))

    #     # Return without any encryption
    #     self.connection.sendall(str(dancer_positions).encode())

    def setup_connection(self):
        random.shuffle(self.move_idxs)
        print("No actions for 60 seconds to give time to connect")
        self.timer = threading.Timer(self.timeout, self.send_timestamp)
        self.timer.start()

        # Wait for a connection
        print('waiting for a connection')
        self.connection, client_address = self.socket.accept()

        print("Enter the secret key: ")
        secret_key = sys.stdin.readline().strip()

        print('connection from', client_address)
        if len(secret_key) == 16 or len(secret_key) == 24 or len(secret_key) == 32:
            pass
        else:
            print("AES key must be either 16, 24, or 32 bytes long")
            self.stop()
        
        return client_address, secret_key # forgot to return the secret key

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()

    # def set_next_action(self):
    #     self.timer.cancel()
    #     if self.has_no_response:  # If no response was sent
    #         self.write_move_to_logger("None", "None", 0)
    #         print("ACTION TIMEOUT")
    #         self.send_dancer_positions() # send dancer positions even at timeout

    #     if self.idx < self.n_moves:
    #         index = self.move_idxs[self.idx]
    #     else:
    #         index = self.n_moves - 1
    #     self.action = self.actions[int(index/NUM_MOVE_PER_ACTION)] # produces indexing error if unchanged
    #     position = random.randrange(0, len(POSITIONS))
    #     self.dancer_positions = POSITIONS[position]
    #     self.idx += 1
    #     self.action_set_time = time.time()

    #     self.timer = threading.Timer(self.timeout, self.set_next_action)
    #     self.has_no_response = True
    #     self.timer.start()

#     def write_move_to_logger(self, predicted_position, predicted_action, sync):
#         log_filepath = self.log_filepath
# #        pos_string = ' '.join(self.dancer_positions)
#         pos_string = self.dancer_positions
#         if not os.path.exists(log_filepath):  # first write
#             with open(log_filepath, 'w') as f:
#                 self.df.to_csv(f)

#         with open(log_filepath, 'a') as f:
#             data = dict()
#             data['timestamp'] = time.time()
#             data['position'] = predicted_position
#             data['action'] = predicted_action
#             data['gt_position'] = pos_string  
#             data['gt_action'] = self.action

#             data['response_time'] = data['timestamp'] - self.action_set_time
#             data['sync'] = sync
#             data['is_action_correct'] = (self.action == predicted_action)
#             data['is_position_correct'] = (pos_string == predicted_position) 

#             self.df = pd.DataFrame(data, index=[0])[self.columns].set_index('timestamp')
#             self.df.to_csv(f, header=False)


# def add_display_label(display_window, label):
#     display_label = Label(display_window, text=str(label))
#     display_label.config(font=('times', 130, 'bold'))
#     display_label.pack(expand=True)
#     return display_label


def main():
    # if len(sys.argv) != 4:
    #     print('Invalid number of arguments')
    #     print('python server.py [IP address] [Port] [groupID]')
    #     sys.exit()

    # ip_addr = sys.argv[1]
    # port_num = int(sys.argv[2])
    # group_id = sys.argv[3]

    # my_server = Server(ip_addr, port_num, group_id)
    # my_server.start()

    dancer_server0 = Server(IP_ADDRESS, PORT_NUM[0], GROUP_ID)
    # dancer_server1 = Server(IP_ADDRESS, PORT_NUM[1], GROUP_ID)
    # dancer_server2 = Server(IP_ADDRESS, PORT_NUM[2], GROUP_ID)

    dancer_server0.start()
    print("dancer_server0 started: IP address:" + IP_ADDRESS + " Port Number: " + str(PORT_NUM[0]) + " Group ID number: " + str(GROUP_ID))

    # dancer_server1.start()
    # print("dancer_server1 started: IP address:" + IP_ADDRESS + " Port Number: " + str(PORT_NUM[1]) + " Group ID number: " + str(GROUP_ID))

    # dancer_server2.start()
    # print("dancer_server1 started: IP address:" + IP_ADDRESS + " Port Number: " + str(PORT_NUM[2]) + " Group ID number: " + str(GROUP_ID))

    # display_window = Tk()
    # action_display = add_display_label(display_window, label=str(my_server.action))
    # position_display = add_display_label(display_window, label=str(my_server.position))
    # display_window.update()
    # while my_server.idx <= my_server.n_moves + 1 and not my_server.shutdown.is_set():  # Display new task
    #     if my_server.idx == my_server.n_moves + 1:
    #         action_display.config(text=str(my_server.idx) + ":" + 'logout')
    #         position_display.config(text=' '.join(my_server.dancer_positions))
    #         if my_server.logout is True:
    #             break
    #     else:
    #         action_display.config(text=str(my_server.idx) + ":" + str(my_server.action))
    #         position_display.config(text=' '.join(my_server.dancer_positions))
    #     display_window.update()
    #     time.sleep(0.2)


if __name__ == '__main__':
    main()


