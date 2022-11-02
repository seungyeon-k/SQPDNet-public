import socket
import time
import pickle
import struct

class Talker():
    def __init__(self, tcp_ip, tcp_port):
        self.BUFFER_SIZE = 368640
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port

        self.socket = None
        self.conn   = None

        self.isOpen = False
        self.payload_size = struct.calcsize("L")
        self.data = bytearray()

    def conn_server(self):
        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.connect((self.tcp_ip , self.tcp_port))
                print("[Robot => UI] connection success")

                self.isOpen = True
                break
            except socket.error as e:
                print("Cannot open socket")
                time.sleep(1)
                continue

    def send_vision(self, image):
        data = pickle.dumps(image)

        # Send message length first
        message_size = struct.pack("L", len(data))

        # Then data
        self.socket.sendall(message_size + data)

    def recv_grasp(self, dict_type=False):
        
        if not self.isOpen:
            self.conn_server()
        
        while len(self.data) < self.payload_size:
            self.data.extend(self.socket.recv(4096))

        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        # Retrieve all data based on message size
        while len(self.data) < msg_size:
            self.data.extend(self.socket.recv(4096))

        grasp_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # Extract frame
        grasp = pickle.loads(grasp_data)
        if dict_type:
            grasp = {k.encode('ascii'): v for k, v in grasp.items()}
        return grasp

    def close_connection(self):
        if self.isOpen:
            self.socket.shutdown(socket.SHUT_RDWR)
            time.sleep(1)
            self.socket.close()
            self.isOpen = False
	
    

    
