from threading import Thread
from SocketServer import SocketServer
from Context import Context

class RecognizeTestThread:
	def __init__(self, socketServer):
		self.runRecognizer = True
		self.socketServer = socketServer

	def Start(self):
		Thread(target=self.recognizer, args=()).start()

	def recognizer(self):
		while self.runRecognizer:
			prompt = 'Enter commands:\n'
			# prompt += '| BACK | HOME | SCREENSHOT | WIFI | SOUND | FLASHLIGHT | NOTIFICATION | RECENT | BLUETOOTH | LOCK | \n'
			# if Context.state == Context.STATE_HOME:
			# 	prompt += '| WECHAT | BROWSER | CAMERA | ALIPAY | MUSIC | TAOBAO | MAIL | WEIBO | CLOCK | MAP |\n'
			# elif Context.state == Context.STATE_WECHAT:
			# 	prompt += '| MOTENTS | SEARCH | ADD | POST | SCANCODE | LIKE | SHOWCODE | COLLECTION\n'
			# elif Context.state == Context.STATE_TEXT:
			# 	prompt += '| UNDO | REDO | LEFT | RIGHT | COPY | CUT | PASTE | BOLD | HIGHLIGHT |\n'
			userInput = input(prompt)
			self.socketServer.SendToAllConnections(userInput + '\n')