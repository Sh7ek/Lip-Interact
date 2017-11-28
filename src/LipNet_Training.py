from src.LipNet import LipNet

lipnet = LipNet(img_c=3, img_w=100, img_h=80, frames_n=70, output_n=11)

lipnet.summary()

lipnet.compile()

