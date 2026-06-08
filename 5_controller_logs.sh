#!/bin/bash

if [ -f "controller.txt" ]
then
    tail -f controller.log
else
    echo "No controller.log file found."
fi