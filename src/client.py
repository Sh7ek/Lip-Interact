import socket
import sys
from time import sleep

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Connect the socket to the port where the server is listening
server_address = ('192.168.1.169', 6000)
sock.connect(server_address)

try:
	# Send data
	while True:
		message = raw_input("Enter lip command: ") + "\n"
		sock.send(message)
		# sleep(1)

finally:
	print >> sys.stderr, 'closing socket'
	sock.close()
