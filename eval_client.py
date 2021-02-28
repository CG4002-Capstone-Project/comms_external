# The client file that is used from the Ultra96 to the evaluation server connection
# For now the client will only send certain format of message.

import os
import sys
import random
import time

import socket
import threading

import base64
import numpy as np
from tkinter import Label, Tk
import pandas as pd
from Crypto.Cipher import AES
from Crypto import Random

# Week 13 test: 8 moves, so 33 in total = (8*4) + 1 (logout)
#ACTIONS = ['muscle', 'weightlifting', 'shoutout', 'dumbbells', 'tornado', 'facewipe', 'pacman', 'shootingstar']
# Week 10 test: 3 moves, repeated 4 times each = 12 moves.
ACTIONS = ['muscle', 'weightlifting', 'shoutout']
POSITIONS = ['1 2 3', '3 2 1', '2 3 1', '3 1 2', '1 3 2', '2 1 3']
LOG_DIR = os.path.join(os.path.dirname(__file__), 'evaluation_logs')
NUM_MOVE_PER_ACTION = 4
N_TRANSITIONS = 6
MESSAGE_SIZE = 3 # position, 1 action, sync 

ENCRYPT_BLOCK_SIZE = 16


class Client(threading.Thread):
    def __init__(self, ip_addr, port_num, group_id, key):
        super(Client, self).__init__()

        # # setup moves
        # self.actions = ACTIONS
        # self.position = POSITIONS 
        # self.n_moves = len(ACTIONS) * NUM_MOVE_PER_ACTION

        # # the moves should be a random distribution
        # self.move_idxs = list(range(self.n_moves)) * NUM_MOVE_PER_ACTION
        # assert self.n_moves == len(self.actions) * NUM_MOVE_PER_ACTION
        # self.action = None
        # self.action_set_time = None

        self.idx = 0
        self.timeout = 60
        self.has_no_response = False
        self.connection = None
        self.timer = None
        self.logout = False

        self.group_id = group_id
        self.key = key

        self.dancer_positions = ['1', '2', '3']

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)


        print('Start connecting... server address: %s port: %s' % server_address, file=sys.stderr)
        self.socket.connect(server_address)
        print('Connected')

    #To encrypt the message, which is a string
    def encrypt_message(self, message):
        raw_message =  "#" + message
        print("raw_message: "+raw_message)
        padded_raw_message = raw_message + ' '* (ENCRYPT_BLOCK_SIZE-(len(raw_message)%ENCRYPT_BLOCK_SIZE))
        print("padded_raw_message: " + padded_raw_message)
        iv = Random.new().read(AES.block_size)
        secret_key = bytes(str(self.key), encoding="utf8")
        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        encrypted_message = base64.b64encode(iv + cipher.encrypt(bytes(padded_raw_message, "utf8")))
        print("encrypted_message: ", encrypted_message)
        return encrypted_message

    #To send the message to the sever
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

def main():
    if len(sys.argv) != 5:
        print('Invalid number of arguments')
        print('python server.py [IP address] [Port] [groupID] [key]')
        sys.exit()

    ip_addr = sys.argv[1]
    port_num = int(sys.argv[2])
    group_id = sys.argv[3]
    key = sys.argv[4]

    my_client = Client(ip_addr, port_num, group_id, key)

    index = 0

    time.sleep(15)

    while True:
        if index == 0:
            my_client.send_message("1 2 3" + '|' + "start" + '|' + "1.5" + '|')
        dancer_position = my_client.receive_dancer_position()
        print("dancer_position: " + dancer_position)
        my_client.send_message("1 2 3" + '|' + "muscle" + '|' + "1.5" + '|')
        
        print("Received dancer postions: ", str(dancer_position))
        time.sleep(4)
        index += 1
        if index == 60:
            my_client.send_message(dancer_position + '|' + "logout" + '|' + "1.5" + '|')
            my_client.stop()

   


if __name__ == '__main__':
    main()

