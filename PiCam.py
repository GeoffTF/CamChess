import socket
from picamera import PiCamera
from io import BytesIO
from time import sleep

host = 'raspberrypi.local'       # Server IP address.
port = 60000                     # Reserve a port for the server.

# Initialise the Raspberry Pi camera.
camera = PiCamera()
camera.resolution = (540, 540)
camera.zoom = (0.25, 0.25, 0.5, 0.5)
camera.exposure_compensation = 6
camera.rotation = 270
print('Camera initialised')

with socket.socket() as sock:        # Create a TCP/IP socket.
    sock.bind((host, port))          # Bind the socket to the port.
    sock.listen()                    # Wait for client connection.
    print('Server listening....')

    conn, addr = sock.accept()       # Establish connection with client.
    print('Got connection from', addr)

    while True:
        # Receive the command to take a picture.
        command = conn.recv(128).decode('UTF-8')
        print('Received the command:', command)

        # Capture a camera image to a BytesIO stream.
        stream = BytesIO()
        camera.capture(stream, 'jpeg')
        print('Camera image captured')

        # Send the number of data bytes to be sent over the connection.
        n2send = stream.tell()
        conn.send(str(n2send).encode('UTF-8'))
        print('Number of bytes to be sent:', n2send)

        # Send the image over the connection.
        stream.seek(0)
        while True:
           chunk = stream.read(4096) # Read a block from the stream.
           if not chunk: break
           conn.send(chunk)          # Send it over the connection.
        print('Image sent')
