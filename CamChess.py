import cv2, chess, chess.engine, chess.pgn
import numpy as np
import socket, io, time, pathlib
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import ImageTk, Image, ImageDraw
from datetime import date

TITLE = 'CamChess'
PATH = pathlib.Path.home().joinpath('CamChess')
GAMES_PATH = PATH.joinpath('Chess_Games.txt')
PIECES_PATH = PATH.joinpath('Pieces')
ENGINE_PATH = '/usr/games/stockfish'

SQSIZE = 50 # Chess board square size in pixels.
BDSIZE = 8 * SQSIZE # Board size in pixels.
D = int((SQSIZE+19) / 20) # Width of border to be omitted from squares.
MIN_AREA = SQSIZE*SQSIZE // 15 # Minimum area for a piece image.

# Create a mask for the largest circular region within a square
# minus its border. The mask is 255 within the circle and 0 outside it.
DIM = SQSIZE-2*D
MASK = np.zeros((DIM, DIM), np.uint8)
CENTRE = (DIM-1) / 2
RSQ = ((DIM-1) / 2)**2
print('DIM', DIM, 'CENTRE', CENTRE, 'RSQ', RSQ)
for y in range(DIM):
    for x in range(DIM):
        if (x-CENTRE)**2 + (y-CENTRE)**2 < RSQ:
            MASK[y,x] = 255
count = MASK[MASK == 0].size # Number of pixels ouside the circle.
MULT = DIM*DIM/count
print('Number outside circle', count, 'MULT', MULT)

White_bottom = True # White at the bottom of diagrams.
START_POSN = ['W']*16 +['X']*32 + ['B']*16
PIECES = {} # Load the chess piece images.
for ch in 'pnbrqkPNBRQK':
    PIECES[ch] = Image.open(PIECES_PATH.joinpath('%s.png' % ch))

board = chess.Board() # Create a python-chess board.
engine_on = False
engine_move = chess.Move.null()
# Set up the chess engine.
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

# Print the engine options.
print('Engine Option         Type    Default'\
      '               Min    Max     Var')
eo = engine.options
def show_empty(x): return x if x != '' else "''"
for k in eo:
    option = str(k).ljust(22) + str(eo[k].type).ljust(8) + \
        show_empty(str(eo[k].default)).ljust(22) + \
        str(eo[k].min).ljust(7) + str(eo[k].max).ljust(8) + \
        ' '.join([str(v) for v in eo[k].var])
    print(option)

close_enabled = True # Main window close button enabled.

def Get_Image():
    ''' Connect to the Raspberry Pi Zero camera server, receive an
    encoded image over the connection, and close the connection.
    Return an OpenCV image.
    '''
    global close_enabled
    host = 'raspberrypi.local'      # Server host name or IP address.
    port = 60000                    # Reserve a port.
    t1 = time.time()
    try:
        with socket.socket() as sock:
            sock.connect((host, port))
            print('Connected to', host, 'port', port)
            # Receive the image over the connection.
            stream = io.BytesIO()
            while True:
                chunk = sock.recv(2048)
                if not chunk: break
                stream.write(chunk)
        t2 = time.time()
        print('Received the image', stream.tell(), 'bytes',
                            t2-t1, 'seconds')
        stream.seek(0)
        image = cv2.imdecode(np.frombuffer(stream.read(), np.uint8),\
                             cv2.IMREAD_COLOR)
    except:
        print('Image Capture Failed')
        image = None
    return image

def Find_Corners(image):
    ''' Find approximate coordinates for the four outer corners of the
    chess board from an image of the empty board. The board is assumed
    to be a standard vinyl roll-up board, as used by chess clubs.
    '''
    # Convert the image to gray scale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the 49 inner corners of the chess board.
    found, in_corners = cv2.findChessboardCorners(gray, (7,7),
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_ADAPTIVE_THRESH)
    if not found: return None
    in_corners = in_corners.reshape((49,2))
    # Add the x and y values for each inner corner. The top-left corner
    # will have the smallest sum, and the bottom-right corner the
    # largest.
    xpy = in_corners.sum(axis = 1)
    inTL = in_corners[np.argmin(xpy)]
    inBR = in_corners[np.argmax(xpy)]
    # Subtract the y value from the x value for each inner corner. The
    # top-right corner will have the smallest difference, and the
    # bottom-left corner the largest.
    xmy = np.diff(in_corners, axis = 1)
    inTR = in_corners[np.argmin(xmy)]
    inBL = in_corners[np.argmax(xmy)]
    print('Inner corners TL', inTL, 'TR', inTR, 'BR', inBR, 'BL', inBL)

    def out_corner(p1, p2):
        # Find the approximate location of one of the four outer
        # corners of the board.
        # p2 is the inner corner nearest to the required outer corner.
        # p1 is the inner corner farthest from the required outer
        # corner.
        # (p3x, p3y) is the approximate location.
        p3x = round(p2[0] + (p2[0] - p1[0])/6)
        p3y = round(p2[1] + (p2[1] - p1[1])/6)
        return p3x, p3y

    # Find approximate locations for the four outer corners.
    outTL, outTR = out_corner(inBR, inTL), out_corner(inBL, inTR)
    outBR, outBL = out_corner(inTL, inBR), out_corner(inTR, inBL)
    print('Approx outer corners', outTL, outTR, outBR, outBL)
    return outTL, outTR, outBR, outBL

def Transform_Image(Corners, image):
    ''' Tranform an image of the chess board to a cropped
    square BDSIZE x BDSIZE image. The inputs are approximate
    coordinates for the four corners of the board and a gray
    scale image of the board. The function returns the transformed
    image and accurate coordinates for the four corners.
    '''
    # Convert the image to gray scale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def find_corner(x, y):
        # Find accurate coordinates for a corner of the board.
        # Construct a Region Of Interest around the approximate
        # location.
        x, y = round(x), round(y)
        d = SQSIZE // 10
        try:
            roi = gray[y-d:y+d+1, x-d:x+d+1]
            # Find the strongest corner in the Region Of Interest.
            qcorners = cv2.goodFeaturesToTrack(roi, 1, 0.1, d)
        except:
            raise ValueError
        if qcorners is None: raise ValueError
        else:
            qcorner = qcorners[0,0,0]+x-d, qcorners[0,0,1]+y-d
        return qcorner

    try:
        QCorners = [find_corner(p[0], p[1]) for p in Corners]
    except ValueError:
        return None, Corners
    #print('Accurate outer corners', QCorners)
    # Construct the prespective transformation matrix.
    pts1 = np.float32(QCorners)
    s = BDSIZE-1
    pts2 = np.float32([[0,0], [s,0], [s,s], [0,s]])
    Matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Transform the image.
    board_image = cv2.warpPerspective(image, Matrix, (BDSIZE,BDSIZE))
    return board_image, QCorners

def Optimise_Thresholds(board_image):
    ''' Use a cropped square image of the start position to optimise
    four threshold values that will be used to identify other positions.
    The blue, green and red channels are used to optimise the threshold
    values. These thresholds identify White pieces on Black squares,
    Black pieces on Black squares, White pieces on White squares, and
    descriminate between Black and White pieces on white squares. These
    thresholds are converted into threshold ratios by dividing them by
    the mean green channel value in the corners of the square concerned.
    This refinement enables the threshold values to adapt to varying
    light levels accross the board, and changes in these levels. The
    function returns the four threshold ratios.
    '''
    # Sets of python-chess square numbers for the start position.
    # White pieces on White squares is sqWoW etc.
    sqWoW = {1,3,5,7,8,10,12,14}
    sqWoB = {0,2,4,6,9,11,13,15}
    sqBoW = {49,51,53,55,56,58,60,62}
    sqBoB = {48,50,52,54,57,59,61,63}
    # Split the board image into the blue, green and red channels.
    (blue, green, red) = cv2.split(board_image)

    def thresh_ratio(gray, squares):
        # Find the optimum threshold ratio for a set of squares for
        # a colour channel.
        n_squares = len(squares)
        thresh_ratio_sum = 0
        for square in squares:
            x, y = square % 8, 7 - square // 8
            x1, x2 = x*SQSIZE+D, (x+1)*SQSIZE-D
            y1, y2 = y*SQSIZE+D, (y+1)*SQSIZE-D
            # Find the mean green value in the corners of the Region
            # Of Interest (ROI), i.e. the square minus a margin.
            roiG = green[y1:y2, x1:x2]
            masked = np.bitwise_and(roiG, 255-MASK)
            mean_cnrs = np.mean(masked)*MULT
            # Find the optimum threshold value in the ROI.
            ROI = gray[y1:y2, x1:x2]
            tv, mask = cv2.threshold(ROI, 0, 255, cv2.THRESH_OTSU)
            thresh_ratio_sum += tv / mean_cnrs
        #print('tv', tv, 'mean', mean_cnrs)
        return thresh_ratio_sum / n_squares

    # Find the threshold ratio for White pieces on Black squares
    # in red light. White pieces should have red values greater
    # than those of the Black squares.
    trWoB = thresh_ratio(red, sqWoB)
    print('trWoB', trWoB)
    # Find the threshold ratio for Black pieces on Black squares
    # in green light. Black pieces should have green values less
    # than those of the Black (actually green) squares.
    trBoB = thresh_ratio(green, sqBoB)
    print('trBoB', trBoB)
    # Find the threshold ratio for White pieces on White squares
    # in blue light. Both White and Black pieces should have blue
    # values less than those of the White squares.
    trWoW = thresh_ratio(blue, sqWoW)
    print('trWoW', trWoW)

    def piece_red_val(squares):
        # For a each member of a set of White squares, find the pixels
        # with a blue value less than the threshold value. For
        # these pixels, find the mean red value for each square.
        # Divide these values by the mean green channel value within
        # within the square (minus a margin) but outside the largest
        # circle within that square. Find the (unweighted) mean
        # value of these ratios.
        n = meanR_sum = 0
        for square in squares:
            x, y = square % 8, 7 - square // 8
            x1, x2 = x*SQSIZE+D, (x+1)*SQSIZE-D
            y1, y2 = y*SQSIZE+D, (y+1)*SQSIZE-D
            # Find the mean green value in the corners of the Region
            # Of Interest (ROI), i.e. the square minus a margin.
            roiG = green[y1:y2, x1:x2]
            masked = np.bitwise_and(roiG, 255-MASK)
            mean_cnrs = np.mean(masked)*MULT
            # Calculate the threshold value from the threshold ratio
            # and the mean green value.
            tvWoW = trWoW * mean_cnrs
            # Make a bit mask for all the pixels with a blue value
            # that is less than the threshold value. (The White pieces
            # on White squares show up most clearly in blue light.)
            roiB = blue[y1:y2, x1:x2]
            _, thresh_mask = cv2.threshold(roiB, tvWoW, 255,\
                                            cv2.THRESH_BINARY_INV)
            # Mask out any pixels that fall outside the central region.
            thresh_mask = np.bitwise_and(thresh_mask, MASK)
            # Find the corresponding pixels in the red channel.
            # (The Black pieces are most easily distinguished from
            # White pieces in red light.)
            roiR = red[y1:y2, x1:x2]
            indices = np.where(thresh_mask == 255)
            filteredR = roiR[indices]
            # Find the pixel count to avoid a posible divide by zero,
            # and calculate the threshold ratio.
            count = filteredR.size
            if count > 0:
                n += 1
                meanR_ratio = np.mean(filteredR) / mean_cnrs
                meanR_sum += meanR_ratio
        return 0 if n == 0 else meanR_sum / n

    # Determine the ratio trWB for descriminating between
    # Black and White pieces. The threshold is taken to be the
    # mid-point between ratios for the mean red values for Black
    # and White pieces.
    red_ratio_W = piece_red_val(sqWoW)
    red_ratio_B = piece_red_val(sqBoW)
    trWB = (red_ratio_W + red_ratio_B) / 2
    print('red_ratio_W', red_ratio_W, 'red_ratio_B', red_ratio_B,\
          'trWB', trWB)
    return trWoB, trBoB, trWoW, trWB

def Identify_Position(board_image):
    ''' Use a cropped square image of the board and the four
    threshold ratios to identify a position. The output is a list
    that gives the occupancy of each square: W for a White
    piece, B for a Black piece and X for an empty square.
    '''
    # Split the board image into the blue, green and red channels.
    (blue, green, red) = cv2.split(board_image)
    posn = []
    for square in range(64):
        x, y = square % 8, 7 - square // 8
        x1, x2 = x*SQSIZE+D, (x+1)*SQSIZE-D
        y1, y2 = y*SQSIZE+D, (y+1)*SQSIZE-D
        # Find the mean green value in the corners of the Region
        # Of Interest (ROI), i.e. the square minus a margin.
        roiG = green[y1:y2, x1:x2]
        masked = np.bitwise_and(roiG, 255-MASK)
        mean_cnrs = np.mean(masked)*MULT
        if (x+y) % 2 == 1: # Black square.
            # Look for a White piece.
            # Calculate the threshold value from the threshold ratio
            # and the mean green value.
            tvWoB = trWoB * mean_cnrs
            # Make a bit mask for all the pixels with a red value
            # that is more than the threshold value. (White pieces
            # on Black squares show up most clearly in red light.)
            roiR = red[y1:y2, x1:x2]
            _, thresh_mask = cv2.threshold(roiR, tvWoB, 255,\
                                            cv2.THRESH_BINARY)
            # Mask out any pixels that fall outside the central region.
            thresh_mask = np.bitwise_and(thresh_mask, MASK)
            # Identify a White piece if the pixel count is more than
            # the minimum.
            count = thresh_mask[thresh_mask == 255].size
            if count >= MIN_AREA:
                posn.append('W')
                continue
            # Look for a Black piece.
            # Calculate the threshold value from the threshold ratio
            # and the mean green value.
            tvBoB = trBoB * mean_cnrs
            # Make a bit mask for all the pixels with a green value
            # that is less than the threshold value. (Black pieces
            # on Black squares show up most clearly in green light.)
            _, thresh_mask = cv2.threshold(roiG, tvBoB, 255,\
                                            cv2.THRESH_BINARY_INV)
            # Mask out any pixels that fall outside the central region.
            thresh_mask = np.bitwise_and(thresh_mask, MASK)
            # Identify a Black piece if the pixel count is more than
            # the minimum, and an empty square otherwise.
            count = thresh_mask[thresh_mask == 255].size
            if count >= MIN_AREA:
                posn.append('B')
                continue
            else: # Piece not found. Empty square.
                posn.append('X')
        else: # White square. Look for a piece.
            # Calculate the threshold value from the threshold ratio
            # and the mean green value.
            tvWoW = trWoW * mean_cnrs
            # Make a bit mask for all the pixels with a blue value
            # that is less than the threshold value. (White pieces
            # on White squares show up most clearly in blue light.
            # Black pieces will show up even more clearly.)
            roiB = blue[y1:y2, x1:x2]
            _, thresh_mask = cv2.threshold(roiB, tvWoW, 255,\
                                            cv2.THRESH_BINARY_INV)
            thresh_mask = np.bitwise_and(thresh_mask, MASK)
            # Mask out any pixels that fall outside the central region.
            count = thresh_mask[thresh_mask == 255].size
            if count >= MIN_AREA: # Piece found. Check colour.
                # Calculate the threshold value from the threshold
                # ratio and the mean green value.
                tvWB = trWB * mean_cnrs
                # Find the corresponding pixels in the red channel.
                # (The Black pieces are most easily distinguished from
                # White pieces in red light.)
                roiR = red[y1:y2, x1:x2]
                indices = np.where(thresh_mask == 255)
                # Find the mean red value of these pixels.
                meanR = np.mean(roiR[indices])
                # Identify a White piece if the mean value is more
                # than the threshold, and a Black piece otherwise.
                posn.append('W' if meanR >= tvWB else 'B')
            else: # Piece not found. Empty square.
                posn.append('X')
    return posn

def Get_Posn(board):
    ''' Get the X, W, B list for the current position from python-chess.
    '''
    posn = []
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            posn.append('X')
        else:
            posn.append('W' if piece.color else 'B')
    return posn

def Promotion_Piece():
    ''' Prompt the user for the promotion piece.
    '''
    global close_enabled, prom_piece
    close_enabled = False # Disable the main window close button.
    child=tk.Toplevel(root)
    child.title('ZeroCam Chess')
    child.lift(aboveThis=root)
    child.title('Promotion')
    child.geometry('160x160')

    def On_Close(): pass

    def On_Q(event=None):
        global prom_piece
        prom_piece = chess.QUEEN
        child.destroy()

    def On_R(event=None):
        global prom_piece
        prom_piece = chess.ROOK
        child.destroy()

    def On_B(event=None):
        global prom_piece
        prom_piece = chess.BISHOP
        child.destroy()

    def On_N(event=None):
        global prom_piece
        prom_piece = chess.KNIGHT
        child.destroy()

    pfont = ('Liberation', 24)
    Qbutton = tk.Button(child, bd=3, text='Q', font=pfont, command=On_Q)
    Qbutton.place(relx=0, rely=0, relheight=0.5, relwidth=0.5)
    Rbutton = tk.Button(child, bd=3, text='R', font=pfont, command=On_R)
    Rbutton.place(relx=0.5, rely=0, relheight=0.5, relwidth=0.5)
    Bbutton = tk.Button(child, bd=3, text='B', font=pfont, command=On_B)
    Bbutton.place(relx=0, rely=0.5, relheight=0.5, relwidth=0.5)
    Nbutton = tk.Button(child, bd=3, text='N', font=pfont, command=On_N)
    Nbutton.place(relx=0.5, rely=0.5, relheight=0.5, relwidth=0.5)
    Qbutton.focus_set()
    child.bind('q', On_Q)
    child.bind('r', On_R)
    child.bind('b', On_B)
    child.bind('n', On_N)
    child.protocol("WM_DELETE_WINDOW", On_Close)
    child.grab_set()
    root.wait_window(child)
    close_enabled = True # Re-enable the main window close buton.

def Identify_Move(changed_squares, board):
    ''' Identify the move from the changed squares and the position
    before the move. Return the move if it is legal. Otherwise return
    a null move. The method used is to deduce the move on the
    assumption that it is legal, and then check that it is indeed
    legal.
    '''
    global prom_piece
    squares = sorted(changed_squares)
    WP = chess.Piece(chess.PAWN, chess.WHITE)
    BP = chess.Piece(chess.PAWN, chess.BLACK)
    move = chess.Move.null()
    if len(squares) == 2: # Simple move, capture or promotion.
        mv = chess.Move.null()
        if board.turn == chess.WHITE: # Check for White promotion.
            if board.piece_at(squares[0]) == WP and squares[1] > 55:
                Promotion_Piece()
                mv = chess.Move(squares[0], squares[1], prom_piece)
        else: # Check for Black promotion,
            if board.piece_at(squares[1]) == BP and squares[0] < 8:
                Promotion_Piece()
                mv = chess.Move(squares[1], squares[0], prom_piece)
        if not mv: # Not a promotion.
            mv = chess.Move(squares[0], squares[1])
            if not mv in board.legal_moves:
                mv = chess.Move(squares[1], squares[0])
    elif len(squares) == 3: # En passant capture.
        if board.turn == chess.WHITE: # White move.
            if squares[1] == squares[2] - 8: # Taken black pawn square.
                mv = chess.Move(squares[0], squares[2])
            else:
                mv = chess.Move(squares[1], squares[2])
        else: # Black move.
            if squares[1] == squares[0] + 8: # Taken white pawn square.
                mv = chess.Move(squares[2], squares[0])
            else:
                mv = chess.Move(squares[1], squares[0])
    else: # Castling move.
        if squares[0] == 4: # White king side castling.
            mv = chess.Move(4, 6)
        elif squares[0] == 0: # White queen side castling.
            mv = chess.Move(4, 2)
        elif squares[0] == 60: # Black king side castling.
            mv = chess.Move(60, 62)
        else: # Black queen side castling.
            mv = chess.Move(60, 58)
    if mv in board.legal_moves: move = mv
    return move

def Show_Position(board, changed_squares=set()):
    ''' Display a chess diagram image showing the position on the
    python-chess board, highlighting any changed squares.
    '''
    sqsize = 64
    bdsize = 8*64
    RED = (255, 70, 70) # RGB light red.
    # Create a light grey bdsize x bdsize image.
    img = Image.new('RGBA', (bdsize,bdsize), (200,200,200))
    draw = ImageDraw.Draw(img)
    # Draw the dark squares.
    for sq in range(64):
        x, y = sq % 8, sq // 8
        p1 = x * sqsize, y * sqsize
        p2 = p1[0] + sqsize, p1[1] + sqsize
        if (x + y) % 2 == 1:
            draw.rectangle([p1, p2], (0,128,43))
    # Highlight any changed squares.
    for sq in changed_squares:
        if White_bottom:
            p1 = (sq % 8 ) * sqsize, (7 - sq//8) * sqsize
        else:
            p1 = (7 - sq % 8 ) * sqsize, (sq//8) * sqsize
        p2 = p1[0] + sqsize, p1[1] + sqsize
        draw.rectangle([p1, p2], RED)
    # Draw the pieces on the board.
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            piece_img = PIECES[piece.symbol()]
            if White_bottom:
                x, y = (sq % 8)*sqsize, (7 - sq//8)*sqsize
            else:
                x, y = (7 - sq % 8)*sqsize, (sq // 8)*sqsize
            img.paste(piece_img, (x, y), mask=piece_img)
    img = ImageTk.PhotoImage(img)
    posn_label.configure(image=img)
    posn_label.image = img

def Show_Image(image):
    ''' Display an OpenCV image.
    '''
    if len(np.shape(image)) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)
    img = img.resize((512,512))
    img = ImageTk.PhotoImage(img)
    posn_label.configure(image=img)
    posn_label.image = img

def Show_Board_Image(board_image, posn):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for square in range(64):
        x, y = square % 8, 7 - square // 8
        X, Y = x*SQSIZE+18, y*SQSIZE+29
        piece = posn[square]
        cv2.putText(board_image, piece, (X,Y), font, 0.5, (0,0,255),\
                    1, cv2.LINE_AA)
    Show_Image(board_image)

def Show_Message_Wait(message, color='red'):
    ''' Display a message and wait for the Next>> button or Return
    key to be pressed.
    '''
    print(message)
    mess_label.configure(text=message, fg=color)
    root.wait_variable(wait_var) # Wait for Next>> to be pressed.

def Save_Game():
    ''' Ask if the user wants to save the game. If so, write the PGN
    for the game to a file, or append it to the file if it
    already exists.
    '''
    global close_enabled
    if board.move_stack: # If there are moves to save.
        close_enabled = False
        answer = messagebox.askyesno(TITLE,\
                                    'Do you want to save the game?')
        close_enabled = True
        if answer:
            header = date.today().strftime('%d/%m/%Y')
            game = str(chess.pgn.Game().from_board(board)).split('\n')
            game = '\n'.join(game[8:]) # Strip the empty PGN header.
            print(header + '\n' + game)
            with open(GAMES_PATH, 'a') as f:
                f.write(header + '\n' + game + '\n\n')

def Make_Engine_Move():
    ''' Get a move from the engine, make it on the python-chess
    board and display it.
    '''
    global engine, engine_move
    engine_move = engine.play(board, chess.engine.Limit(time=1.0)).move
    changed_squares = {engine_move.from_square, engine_move.to_square}
    # Show the engine move on the chess board diagram.
    Show_Position(board, changed_squares)
    # Show the move.
    mv_str = 'Engine Move: ' + board.variation_san([engine_move])
    board.push(engine_move)
    if board.is_game_over():
        mv_str = mv_str + ' ' + board.result()
    Show_Message_Wait(mv_str, 'black')

def On_Closing():
    ''' The main window close button has been clicked. Close the
    program down if close_enabled is True, otherwise return.
    '''
    global close_enabled
    if close_enabled == False: return
    Save_Game()
    root.destroy()

def On_Next(event=None):
    ''' The Next button (or Return key) has been pressed.
    '''
    wait_var.set(1) # Release the wait in Show_Message.

def On_Rotate(event=None):
    ''' Rotate the board image through 180 degress.
    '''
    global board_image, Corners
    c0 = Corners[0]
    Corners[0] = Corners[2]
    Corners[2] = c0
    c1 = Corners[1]
    Corners[1] = Corners[3]
    Corners[3] = c1
    board_image, Corners = Transform_Image(Corners, image)
    wait_var.set(1) # Release the wait in Show_Message.

def On_Takeback(event=None):
    ''' The Takeback button (or T key) has been pressed. A move should
    have been taken back on the physical chess board. Take back the
    move on the python-chess board.
    '''
    global board
    try:
        move = board.pop()
    except:
        Show_Message_Wait('No Move to Take Back')
        return
    print('Move taken back')
    Show_Position(board)
    mess_label.configure(text='Move Taken Back')

def On_Engine(event=None):
    ''' The Engine button (or E key) has been pressed. If the engine
    is off, turn it on. Highlight the button in pink. Get a move from
    the engine and display it. If the engine is on, turn it off, and
    remove the pink highlighting.
    '''
    global engine_on, engine_color
    if engine_on:
        engine_on = False
        engine_button.configure(bg='light gray',\
                                activebackground='light gray')
    else:
        engine_color = board.turn
        engine_on = True
        engine_button.configure(bg='pink', activebackground='pink')
        # Get a move from the engine, and display it.
        Make_Engine_Move()

def On_Level(event=None):
    global close_enabled
    ''' The Level button (or L key) has been pressed. Ask the user to
    set the engine skill level (if that option is available).
    '''
    if 'Skill Level' in eo:
        min_lv = eo['Skill Level'].min
        max_lv = eo['Skill Level'].max
        mess = 'Set Skill Level (' + str(min_lv) + '-' + \
               str(max_lv) + ')'
        close_enabled = False
        level = simpledialog.askinteger(TITLE, mess,\
                                        minvalue=min_lv,\
                                        maxvalue = max_lv)
        close_enabled = True
        if level:
            print('Skill level', level)
            mess_label.configure(text='Skill Level '+str(level))
            engine.configure({'Skill Level': level})
        next_button.focus_set()

def On_Flip(event=None):
    ''' Rotate chess diagrams through 180 degrees.
    '''
    global White_bottom
    White_bottom = not White_bottom
    Show_Position(board)

# Create a window to display board images, messages and command buttons.
root = tk.Tk()
root.title(TITLE)
wait_var = tk.IntVar()
img = Image.new('RGBA', (512,512), (0,0,0))
img = ImageTk.PhotoImage(img)
posn_frame = tk.Frame(root, width=512, bd=6, relief=tk.FLAT)
posn_frame.pack(side=tk.TOP)
posn_label = tk.Label(posn_frame, image=img)
posn_label.pack()
mess_frame = tk.Frame(root, width=512, bd=4, relief=tk.GROOVE)
mess_frame.pack(fill='both', expand=True)
mess_label = tk.Label(mess_frame, text='Set Up an Empty Board',\
                      height=2, font=('Liberation', 24), pady=8)
mess_label.pack(side=tk.TOP)
button_frame = tk.Frame(root, bd=3)
button_frame.pack(side=tk.TOP)
bfont = ('Liberation', 12)
next_button = tk.Button(button_frame, text='Next>>', font=bfont,\
                        bd=3, command=On_Next)
next_button.pack(side=tk.LEFT)
rotate_button = tk.Button(button_frame, text='Rotate', font=bfont,\
                          bd=3, underline=0, command=On_Rotate)
takeback_button = tk.Button(button_frame, text='Takeback', font=bfont,\
                            bd=3, underline=0, command=On_Takeback)
engine_button = tk.Button(button_frame, text='Engine', font=bfont,\
                          bd=3, underline=0, command=On_Engine)
level_button = tk.Button(button_frame, text='Level', font=bfont,\
                         bd=3, underline=0, command=On_Level)
f_button = tk.Button(button_frame, text='Flip', font=bfont,\
                         bd=3, underline=0, command=On_Flip)
next_button.focus_set()
root.bind('<Return>', On_Next)
root.bind('t', On_Takeback)
root.bind('e', On_Engine)
root.bind('l', On_Level)
root.bind('f', On_Flip)
root.bind('r', On_Rotate)
root.protocol("WM_DELETE_WINDOW", On_Closing)

# Identify the chess board.
while True:
    image = Get_Image()
    if image is None:
        Show_Message_Wait('Image Capture Failed')
        continue
    Corners = Find_Corners(image)
    if Corners is None:
        Show_Image(image)
        Show_Message_Wait('Chess Board Not Found')
        continue
    board_image, Corners = Transform_Image(Corners, image)
    if board_image is not None: break
    Show_Image(image)
    Show_Message_Wait('Chess Board Not Found')
Show_Image(image)
Show_Message_Wait('Chess Board Found\nSet Up Start Position', 'black')
rotate_button.pack(side=tk.LEFT)

# Identify the start position.
while True:
    while True:
        image = Get_Image()
        if image is None:
            Show_Message_Wait('Image Capture Failed')
            continue
        board_image, Corners = Transform_Image(Corners, image)
        if board_image is not None: break
        Show_Image(image)
        Show_Message_Wait('Chess Board Not Found')
    Show_Image(board_image)
    Show_Message_Wait('White at Bottom?\nIf not Rotate', 'black')
    trWoB, trBoB, trWoW, trWB = Optimise_Thresholds(board_image)
    posn = Identify_Position(board_image)
    if posn == START_POSN: break
    print(posn)
    Show_Board_Image(board_image, posn)
    Show_Message_Wait('Start Position Not Found')
rotate_button.pack_forget() # Remove the Rotate button.
root.unbind('r')

# Show the Takeback, Engine, Level and R buttons.
takeback_button.pack(side=tk.LEFT)
engine_button.pack(side=tk.LEFT)
level_button.pack(side=tk.LEFT)
f_button.pack(side=tk.LEFT)

Show_Position(board)
Show_Message_Wait('Make a Move or Press Engine', 'black')

# Respond to moves.
while True:
    if engine_on and engine_color==board.turn and not engine_move:
        # Make a move from the engine, and display it.
        Make_Engine_Move()
    image = Get_Image()
    if image is None:
        Show_Message_Wait('Image Capture Failed')
        break
    board_image, Corners = Transform_Image(Corners, image)
    if board_image is None:
        Show_Image(image)
        Show_Message_Wait('Chess Board Not Found')
        continue
    posn = Identify_Position(board_image)
    #Show_Board_Image(board_image, posn)
    #Show_Message_Wait('Position from Image', 'black')
    if posn == START_POSN:
        # New game: the start position has been set up again.
        mess_label.configure(text='New Game')
        Save_Game() # Ask the user if he wants to save the old game.
        board = chess.Board() # Set up a new game.
        engine_on = False
        engine_button.configure(bg='light gray',\
                                activebackground='light gray')
        Show_Position(board)
        root.wait_variable(wait_var) # Wait for Next>> to be pressed.
        continue
    before_mv_posn = Get_Posn(board) # From the python-chess board.
    diff = {sq for sq in range(64) if posn[sq] != before_mv_posn[sq]}
    if not diff:
        Show_Position(board, set())
        Show_Message_Wait('Move Not Found')
        continue
    move = Identify_Move(diff, board)
    if move == chess.Move.null():
        Show_Position(board, diff)
        Show_Message_Wait('Illegal Move')
        continue
    # Legal move found.
    mv_str = board.variation_san([move])
    color = 'black'
    engine_move = chess.Move.null()
    board.push(move)
    if board.is_game_over():
        mv_str = mv_str + ' ' + board.result()
        color = 'red'
    Show_Position(board, {move.from_square, move.to_square})
    mess_label.configure(text=mv_str, fg=color)
    if not engine_on:
        root.wait_variable(wait_var) # Wait for Next>> to be pressed.

root.mainloop()
