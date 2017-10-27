import cv2
from src.ImageFPS import FPS
from src.ImageCamCaptureThread import WebCamVideoStream

video_stream = WebCamVideoStream(src=0).start()
fps = FPS().start()

# while True:
while True:
	frame = video_stream.read()
	if video_stream.notEmpty():  # cannot be omitted, because the first frame may be empty on the camera
		# Display the resulting frame
		cv2.imshow('frame', frame)
		fps.update()
		# Press Q on keyboard to stop recording
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

video_stream.stop()
cv2.destroyAllWindows()
