#! /bin/bash

#
# delete all pyc files
#
find . -maxdepth 4 -name "*.pyc" -type f -delete

#
# clean and make using catkin
#
catkin_make clean
catkin_make
