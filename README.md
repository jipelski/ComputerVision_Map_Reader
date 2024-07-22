# Map Pointer Detection

This is a computer vision program that identifies the position and orientation of a red triangular pointer on a map. The map sits on a dark blue background, and the coordinate system starts at the bottom left corner with the ranges [0, 1] for both the x and y axes.

## Description

The program processes an image to find the coordinates of the tip of a red triangular pointer placed on a map and calculates the direction in which the pointer is pointing. The direction is measured in degrees clockwise from north. North is determined by the direction to which a green arrow is pointing on the map.

## Features

- Detects a red triangular pointer on a map with a dark blue background.
- Calculates the position of the tip of the pointer.
- Determines the direction (bearing) the pointer is facing in degrees clockwise from north.

## Requirements

- Python 3.6+
- OpenCV
- NumPy

## Usage
To run the program, use the following command:
  python3 mapreader.py <filename>
- '<filename>': The image file to be processed.

## Assumptions and Restrictions
The map should be placed on a dark blue background.
The red triangular pointer should be completely on the map.
The map should be oriented such that the green arrow points upwards.
