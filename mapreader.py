#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# mapreader.py -- solution to finding a red pointer on a map
# ------------------------------------------------------------------------------
# PURPOSE
# This program finds the coordinates of the tip of a red triangular pointer
# placed on a map that sits on a dark blue background. It also calculates the
# direction in which the pointer is pointing counting the degrees clockwise
# from north. The coordinate system for the map starts on the bottom left corner
# of the map and increases from there. The range of the system for the y axis is
# between 0 and 1 and the range for the x axis is between 0 and 1.
# 
#
#
# USAGE
#   python3 mapreader.py <filename>
# where <filename> is the image to be processed.
#
#
# RESTRICTIONS
# The program only works if the image to be processed contains a map on a blue
# background and a red triangular pointer that is placed completely on the map.
# It will be assumed that the map is already oriented so that the green arrow
# points upwards.
#
#
# AUTHOR
# Registration number 1801808
# I hereby certify that this program is entirely my own work.
# ------------------------------------------------------------------------------
import cv2
import math
import numpy
import sys


# ------------------------------------------------------------------------------
# Function definitions
# ------------------------------------------------------------------------------

# The following function locates an object based on its colour and either
# removes it or removes everything but the object.
# The function takes four arguments. The first argument is the image to be
# processed. The image is converted to HSV in order to ease colour
# identification.
# The second and third arguments are lower and upper bounds respectively
# used for colour recognition. Using those values we create two arrays that are
# in turn used to find the pixels who's HSV value is within the specified range,
# thus creating a mask of the object.
# The fourth argument is a boolean called remove_object. If it's true it inverts
# the created mask, otherwise the mask stays the same.
# Finally the created mask is put over the original picture and the created is
# image returned.
#
# I chose to make the following function run each time there is an object to
# be found in the spirit of reusability of code.

def segment_object(image, lower_bound, upper_bound, remove_object):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    low_arr = numpy.array([lower_bound, 50, 30], numpy.uint8)
    upp_arr = numpy.array([upper_bound, 255, 255], numpy.uint8)

    mask = cv2.inRange(image_hsv, low_arr, upp_arr)

    if remove_object:
        mask = 255 - mask

    result = cv2.bitwise_and(image, image, mask=mask)

    return result


# The following function finds the corners of the largest contour in an image.
# The image is previously processed so that it contains only the object
# of interest.
# It takes an image as a single argument. The image is then transformed into
# greyscale and the contours are found and saved in an array.
# The function goes through the array of contours and finds the one with
# the biggest area. We chose the one with the biggest area because it will
# most likely be the contour of our shape. The contour is then approximated
# using Douglas-Peucker algorithm with the precision specified being
# ten percent of the shape's perimeter.
def find_vertices(image):
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, h = cv2.findContours(image_grey, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    area_max = -1
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_max:
            area_max = area
            big_cntr = c

    # The next two lines have been adapted from
    # https://pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    peri = cv2.arcLength(big_cntr, True)
    reshaped_contour = cv2.approxPolyDP(big_cntr, 0.1 * peri, True)

    return reshaped_contour


# The following function has been adapted from
# https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# The function takes the coordinates of a polygon and returns an ordered
# list of said points such that the first point will be the top-left point,
# the second will be the top-right point, the third will be the bottom-right
# point and the fourth will be the bottom-left point
def order_points(points):
    ordered_list = numpy.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    ordered_list[0] = points[numpy.argmin(s)]
    ordered_list[2] = points[numpy.argmax(s)]

    diff = numpy.diff(points, axis=1)
    ordered_list[1] = points[numpy.argmin(diff)]
    ordered_list[3] = points[numpy.argmax(diff)]

    return ordered_list


# The following function computes the distance between 2 points.
# It takes 2 arrays as arguments, uses the formula for finding
# the distance between 2 points in a 2D plane and returns it
def distance_of_points(a, b):
    distance = numpy.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))

    return distance


# The following function has been adapted from
# https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# The function takes an image and a list of points.
# First the list of points is ordered so that we know exactly which point
# we are working with at any given time. Then the maximum width and
# height are calculated. We use these values to create a matrix that will
# serve as our new image. The image is then warped into perspective and
# returned.
def warp_image(image, points):
    rectangle = order_points(points)
    (topLeft, topRight, bottRight, bottLeft) = rectangle

    width_a = distance_of_points(bottRight, bottLeft)
    width_b = distance_of_points(topRight, topLeft)
    max_width = max(int(width_a), int(width_b))

    height_a = distance_of_points(topRight, bottRight)
    height_b = distance_of_points(topLeft, bottLeft)
    max_height = max(int(height_a), int(height_b))

    destination = numpy.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rectangle, destination)
    warped_image = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped_image


# The following function computes the slope of a line given 2 points.
# If the x1 and x2 values are equal add 1 to x2 so the difference will be 1.

def get_slope(x1, y1, x2, y2):
    if x2 == x1:
        x2 = x2 + 1
    slope = (y2 - y1) / (x2 - x1)

    return slope


# ------------------------------------------------------------------------------
# End of function definition
# ------------------------------------------------------------------------------
# Main program.
# ------------------------------------------------------------------------------


# Ensure we were invoked with a single argument.
# If not print a description on how to use the program.

if len(sys.argv) != 2:
    print("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    sys.exit(1)

im_file = sys.argv[1]
im = cv2.imread(im_file)

# Ensure the image exists.
# If not print an error message describing the file could not be found.
if im is None:
    print("File not found")
    sys.exit(1)

# Locate the blue background and segment the map around it.
imageWithMask = segment_object(im, 97, 107, True)

# Find the corners of the map, order them and warp the image into perspective.
vertices = find_vertices(imageWithMask)
ordered_vertices = order_points(vertices.reshape(4, 2))
warped = warp_image(im, ordered_vertices)

# Find the width and height of the map by manually picking up the x value of
# the top right corner and the y value of the bottom right corner of the
# warped map.
warpedPoints = find_vertices(warped)
map_width = warpedPoints[2][0][0]
map_height = warpedPoints[2][0][1]

# Locate the red triangle, segment it and find its corners.
# I couldn't find a cleaner way to look for red values in HSV.
triangle = segment_object(warped, 160, 179, False)
(p1, p2, p3) = find_vertices(triangle)

# Find the lowest edge of the triangle by comparing the lengths of each edge.
# Knowing that the triangle is isosceles the point that is not part of the
# found edge will be the tip of our pointer.
# We will also create a point in the middle of the found edge to help with
# calculating the direction of the pointer.
dist1 = distance_of_points(p1[0], p2[0])
dist2 = distance_of_points(p1[0], p3[0])
dist3 = distance_of_points(p2[0], p3[0])

min_dist = min(dist1, dist2, dist3)

if dist1 == min_dist:
    xpos = p3[0][0]
    ypos = p3[0][1]
    mid_x = (p1[0][0] + p2[0][0]) / 2
    mid_y = (p1[0][1] + p2[0][1]) / 2

elif dist2 == min_dist:
    xpos = p2[0][0]
    ypos = p2[0][1]
    mid_x = (p1[0][0] + p3[0][0]) / 2
    mid_y = (p1[0][1] + p3[0][1]) / 2

else:
    xpos = p1[0][0]
    ypos = p1[0][1]
    mid_x = (p3[0][0] + p2[0][0]) / 2
    mid_y = (p3[0][1] + p2[0][1]) / 2

mid_x = int(mid_x)
mid_y = int(mid_y)

# Calculate the slope of the line between the middle point and north,
# and the slope of the line between the pointer head and the middle point.
north_slope = get_slope(mid_x + 1, mid_y + 1, mid_x + 1, map_width)
pointer_slope = get_slope(xpos, ypos, mid_x + 1, mid_y + 1)

# Calculate the tangent using the slopes of the pointer line and the north line.
# Transform the tangent into radians using the inverse function.
# Transform the radians into degrees.
angle_tangent = (north_slope - pointer_slope) / \
                (1 + north_slope * pointer_slope)

pointer_dirr = abs(math.atan(angle_tangent))
pointer_dirr = math.degrees(pointer_dirr)

# Find the direction of the pointer by using the unit circle chart and the
# position of the pointer relative to the position of the middle point.
# The tangent is negative only in the second quadrant and the fourth quadrant.
# The program starts the coordinate system from the top-left corner.
# Check the relative position of the pointer head with the middle point and add 
# 180 degrees to the pointer_dirr value if the pointer head is in the third 
# quadrant or 270 if it's in the fourth quadrant.
# Due to the found triangle corners not being 100% true with
# actual pointer corners we have to check the tangent to know exactly
# if we are in either third or fourth quadrant and make adjustments to
# the pointer_dirr value accordingly.

if mid_x > xpos:
    if mid_y < ypos:
        pointer_dirr = pointer_dirr + 180
    elif angle_tangent < 0:
        pointer_dirr = pointer_dirr + 270
    else:
        pointer_dirr = pointer_dirr + 180

hdg = pointer_dirr

# As stated before the program counts from the top-left corner of the map,
# so the y value is inversed. To fix this we have to substract the height of
# the map from the y value.
ypos = abs(ypos - map_height)

xpos = xpos / map_width
ypos = ypos / map_height

print("The filename to work on is %s." % sys.argv[1])

# Output the position and bearing in the form required by the test harness.
print("POSITION %.3f %.3f" % (xpos, ypos))
print("BEARING %.1f" % hdg)

# ------------------------------------------------------------------------------
# End of program
# ------------------------------------------------------------------------------
