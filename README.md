![Logo](Images/Logo.png)

CamChess is a free open source program that enables you to play against a computer with a standard tournament board and pieces. It uses a camera to identify moves on the board. It is optimised for a vinyl roll up board with dark green squares and black and "light wood" coloured plastic pieces. It currently uses a Raspberry Pi Zero with an attached ZeroCam to capture board images. The Pi Zero talks to another computer (e.g a faster Raspberry Pi) over TCP/IP. I have set up the Pi Zero as an Ethernet gadget. It both receives power an sends data over a single USB cable. The repository contains PiCam.py which runs on the Pi Zero, and CamChess.py which runs on the other computer. It also contains ImClient.py, which runs on the other computer and is useful for setting up the camera. The programs are written in Python, and CamChess.py uses OpenCV and python-chess.

![Screen Shot](Images/Screen.png)

## Website

The CamChess website can be found [here](https://camchess.blogspot.com).

## Creator

Geoff Fergusson

## Licenses

CamChess (and PiCam) are available under the MIT license. See the LICENCE file for details. CamChess uses Merida chess piece images. These images are available under a GNU licence, which is in the Pieces folder.
