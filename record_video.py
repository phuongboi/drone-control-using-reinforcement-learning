import cv2


video_path = ("/home/samvdh/Videos/vlc-record-2023-12-30-15h04m47s-screen___-.avi")

vid_cap = cv2.VideoCapture(video_path)

frame_width = int(vid_cap.get(3))
frame_height = int(vid_cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('results/hover2.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while(True):
    ret, frame = vid_cap.read()

    if ret != True:
        break

    # Write the frame into the
    # file 'filename.avi'
    result.write(frame)
