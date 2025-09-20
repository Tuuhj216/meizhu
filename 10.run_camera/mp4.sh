#!/bin/sh


OUTPUT_FILE="output.mp4"

#ffmpeg -f v4l2 -i /dev/video0 -vcodec libx264 -pix_fmt yuv420p -preset fast -crf 23 -y "$OUTPUT_FILE"

gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! avenc_h264 ! mp4mux ! filesink location="$OUTPUT_FILE"


