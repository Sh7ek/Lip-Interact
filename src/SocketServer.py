import socket
import sys
from time import sleep
from threading import Thread
from src.Context import Context

class SocketServer:
	def __init__(self, serverIP='192.168.0.102', serverPort=6000):
		self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server_address = (serverIP, serverPort)
		self.runServer = True
		self.connectList = []

	def Start(self):
		Thread(target=self.RunServer, args=()).start()

	def RunServer(self):
		self.serverSocket.bind(self.server_address)
		self.serverSocket.listen(5)
		print('Server Listening ...')

		while self.runServer:
			connection, client_address = self.serverSocket.accept()
			print('Client Connected: {} {}'.format(client_address[0], client_address[1]))
			self.connectList.append((connection, client_address))
			Thread(target=self.MessageReceiver, args=(connection,)).start()

		self.serverSocket.close()

	def MessageReceiver(self, connection):
		while self.runServer:
			data = connection.recv(1024)
			if not data:
				print('Receive Data Error')
				break
			dataString = data.decode('utf-8')
			if '\n' in dataString:
				stateString = dataString.split('\n')[-1]
				if len(stateString) > 0:
					Context.state = int(stateString)

		connection.close()

	def SendToAllConnections(self, message='test'):
		for (connection, client_address) in self.connectList:
			connection.sendto(message.encode(), client_address)

	def StopServer(self):
		self.runServer = False
