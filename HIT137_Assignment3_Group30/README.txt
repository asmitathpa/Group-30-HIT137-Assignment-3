HIT137 ASSIGNMENT 3 - PICTURE PUZZLE
Group-30

PROJECT OVERVIEW

Our project is a desktop picture puzzle built with Python, Tkinter and OpenCV.
It divides an image into tiles and scrambles them using swaps, rotations and
flips. The player restores the picture using the original image as a reference.

GROUP MEMBERS

William Wendl - S361240
Kamala Thapa - S407760
Usha Khadka - S408025

Repository: https://github.com/asmitathpa/Group-30-HIT137-Assignment-3

SETUP AND RUN

1. Extract the ZIP and open a terminal inside the project folder.
2. Install the dependencies:
   python -m pip install -r requirements.txt
3. Start the program:
   python image_puzzle.py
4. Select Try demo or Load image.

Use Python 3.10 or newer. If the computer uses the py launcher, replace python
with py. Tkinter is included with the standard Python Windows installer.

CONTROLS

Left click: select a tile; click a second tile to swap them.
Click the selected tile again: deselect it.
Right click: rotate a tile 90 degrees clockwise.
Shift + left click: flip a tile horizontally.
Control + left click: alternative rotation control.
Hint: mark an incorrect tile and its correct position with blue circles.
Solve: restore the image and reset the move and tiles-left counters.

The original image appears on the left and the playable puzzle on the right.
Green ticks show tiles in the correct position and orientation. Each swap,
rotation or flip counts as one move. Three hints are available per round;
the hint clears after the next move. Completion displays a message and locks
the puzzle. Loading another image starts a new round.

IMAGE PROCESSING

JPG, PNG and BMP images are supported. Images are resized with their aspect
ratio preserved, then padded into a square. Select a 3x3, 4x4 or 5x5 grid before
loading. Tiles have equal dimensions so rotations preserve the board layout.

The initial scramble uses 6, 12 or 20 actions for the respective grid sizes.
Each initial tile is targeted once. The scramble includes swaps, rotations
and flips, generated together before displaying the puzzle.

An optional timed challenge provides 5, 8 or 10 minutes according to grid size.
Grid and timer settings apply to the next loaded image.

PROGRAM STRUCTURE

ImageLoader: reads, resizes and pads images using OpenCV.
Tile: stores a tile's image, original position and orientation.
Transformation: defines a common interface for reversible actions.
SwapTransformation, RotateTransformation and FlipTransformation: implement
that interface through inheritance and polymorphism.
Puzzle: manages the board, scramble, moves, hints and action history.
PuzzleApp: manages the Tkinter interface, mouse events, display and timer.

Tile data is encapsulated in its class. Public image access returns a copy.
The orientation records clockwise quarter-turns and reflection. Solve reverses
the recorded transformations in reverse order to restore the picture.

TESTING

Run the test suite:
python -m unittest -v test_image_puzzle.py

The suite checks image formats, grid sizes, transformations, scrambling,
controls, hints, counters, completion and error handling. GUI tests require
a display. Screenshots and the dated automated test log are in outputs/.

FILES

image_puzzle.py          Application source.
test_image_puzzle.py     Automated tests.
requirements.txt        Dependencies.
assets/demo.png         Demonstration image.
outputs/                Screenshots and test log.
GROUP_DETAILS.txt       Group information.
CONTRIBUTIONS.txt       Team responsibilities.
github_link.txt         Repository URL.
