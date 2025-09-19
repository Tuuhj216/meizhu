#!/bin/sh


gst-launch-1.0 -v v4l2src device=/dev/video0 ! "video/x-raw,format=YUY2,width=1280,height=800" ! queue ! waylandsink
