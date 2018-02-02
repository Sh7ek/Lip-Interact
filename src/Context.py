class Context:
	# Context.state describes the using context on the smartphone
	# state 1: on home screen, users can do system commands and open xxx apps
	# state 2: in wechat app, users can do system commands and wechat operations
	# state 3: in text edit app, user can do system commands and edit operations
	# state 4: other contexts, only allows system commands

	state = 0

	SystemCommands = ['BACK', 'HOME', 'SCREENSHOT', 'WIFI', 'SOUND', 'FLASHLIGHT', 'NOTIFICATION', 'RECENT', 'BLUETOOTH', 'LOCK']
	HomeCommands = ['WECHAT', 'BROWSER', 'CAMERA', 'ALIPAY', 'MUSIC', 'TAOBAO', 'MAIL', 'WEIBO', 'CLOCK', 'MAP']
