from threading import Thread
import cv2

class WebCamVideoStream:
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False
		self.stream.set(3, 640)
		self.stream.set(4, 480)
		print("width: " + str(self.stream.get(3)))
		print("height: " + str(self.stream.get(4)))

	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		while not self.stopped:
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		return cv2.flip(self.frame, 1)
		# return self.frame.copy()

	def notEmpty(self):
		return self.grabbed

	def stop(self):
		self.stopped = True
