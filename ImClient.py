''' This program runs on the main computer and is helpful for
setting up the camera.
'''
import socket, io
import time
import tkinter as tk
from PIL import ImageTk, Image

host = 'raspberrypi.local'      # Server IP address.
port = 60000                    # Reserve a port.

sock = socket.socket()          # Create a TCP/IP socket.
sock.connect((host, port))      # Connect to the server.
print('Connected to', host, 'port', port)

def on_mouse_click(e):
    print('Mouse clicked')
    t1 = time.time()
    # Send the command to take a picture.
    sock.send('SNAP'.encode('UTF-8'))
    print('SNAP command sent')

    # Receive the number of bytes to be sent by the server.
    bytes = sock.recv(128)
    nsent = int(bytes.decode('UTF-8'))
    print('Received number of bytes to be sent:', nsent)

    # Receive the picture over the connection.
    stream = io.BytesIO()
    nreceived = 0
    while True:
        chunk = sock.recv(4096)  # Receive data over the connection.
        stream.write(chunk)      # Write it to the stream.
        nreceived += len(chunk)
        if nreceived == nsent: break
    t2 = time.time()
    print('Received the image', t2-t1, 'seconds')

    # Display the image.
    image = Image.open(stream).convert('RGBA')
    print('PIL image created', image.size)
    stream.close()
    img = ImageTk.PhotoImage(image)
    print('Photo image created')
    label.configure(image=img)
    label.image = img

# Create a window to display images.
# A left mouse click in the window updates the image.
# Closing the window closes the connection with the server.
root = tk.Tk()
root.title('Camera Image')
img = Image.new('RGBA', (540,540), (0,127,0))
img = ImageTk.PhotoImage(img)
label = tk.Label(root, image=img)
label.pack()
root.bind('<Button-1>', on_mouse_click)
root.mainloop()

sock.close()
print('Connection closed')
