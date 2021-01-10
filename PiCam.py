import socket
from picamera import PiCamera
from io import BytesIO
from time import sleep

host = 'raspberrypi.local'        # Server host name or IP address.
port = 60000                      # Reserve a port for the server.

# Initialise the Raspberry Pi camera.
camera = PiCamera()
camera.resolution = (540, 540)
camera.zoom = (0.25, 0.25, 0.5, 0.5)
camera.exposure_compensation = 6
camera.rotation = 270
sleep(2)
print('Camera initialised')

# Create a TCP/IP socket and bind it to the port.
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((host, port))

    while True:
        # Wait for a request to connect.
        sock.listen()
        print('Server listening....')
        conn, addr = sock.accept()
        print('Got connection from', addr)

        # Capture a camera image to a BytesIO stream.
        stream = BytesIO()
        camera.capture(stream, 'jpeg')
        print('Camera image captured')
    
        # Send the image over the connection.
        stream.seek(0)
        with conn:
            while True:
               chunk = stream.read(2048)
               if not chunk: break
               conn.sendall(chunk)
        print('Image sent')
