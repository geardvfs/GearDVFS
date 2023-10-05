from ffpyplayer.player import MediaPlayer
import numpy as np
import cv2
import time


def play_video(file,interval=30):
    fps, cnt, curTime, prevTime = [0]*4
    player = MediaPlayer(file)
    while True:
        frame, val = player.get_frame()
        if val == 'eof': 
            break # this is the difference
        if frame is not None:
            image, pts = frame
            w, h = image.get_size()
            # convert to array width, height
            img = np.asarray(image.to_bytearray()[0]).reshape(h,w,3)
            # convert RGB to BGR because `cv2` need it to display it
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            str_msg="FPS: %0.1f" % fps
            cv2.putText(img,str_msg,(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
            time.sleep(val)
            cv2.imshow('video', img)
            cnt=cnt+1
            if cnt==interval:
                curTime=time.time()
                cnt=0
                sec=curTime-prevTime
                prevTime=curTime
                fps=round(interval/(sec),1)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    cv2.destroyAllWindows()
    player.close_player()

if __name__ == '__main__':
    play_video("../traffic.mp4",30)
