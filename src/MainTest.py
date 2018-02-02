from src.SocketServer import SocketServer
from src.RecognizeTestThread import RecognizeTestThread

if __name__ == '__main__':
	socketServer = SocketServer(serverIP='192.168.1.100', serverPort=6000)
	socketServer.Start()

	recognizeThread = RecognizeTestThread(socketServer=socketServer)
	recognizeThread.Start()
