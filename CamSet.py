import socket, io
import time
import tkinter as tk
from tkinter import simpledialog
from PIL import ImageTk, Image

host = 'raspberrypi.local'      # Server host name or IP address.
port = 60000                    # Reserve a port.

def on_mouse_click(e):
    ''' Connect to the camera server, receive an image over
    the connection and close the connection. Display the image.
    '''
    print('Mouse clicked')
    t1 = time.time()

    with socket.socket() as sock:
        sock.connect((host, port))
        print(time.time()-t1, 'Connected to', host, 'port', port)
        
        # Receive the image over the connection.
        stream = io.BytesIO()
        while True:
            chunk = sock.recv(2048)
            if not chunk: break
            stream.write(chunk)

    t2 = time.time()
    print('Received the image', stream.tell(), 'bytes', t2-t1, 'seconds')

    # Display the image.
    image = Image.open(stream).convert('RGBA')
    print('PIL image created', image.size)
    img = ImageTk.PhotoImage(image)
    print('Photo image created')
    label.configure(image=img)
    label.image = img
    
    stream.close()
  
# Create a window to display images.
# A left mouse click in the window updates the image.
root = tk.Tk()
root.title('Camera Image')
img = Image.new('RGBA', (540,540), (0,127,0))
img = ImageTk.PhotoImage(img)
label = tk.Label(root, image=img)
label.pack()
root.bind('<Button-1>', on_mouse_click)
root.mainloop()
