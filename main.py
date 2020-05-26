print('Importing libraries...')

import argparse
from av import open, VideoFrame
import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from PIL import Image, ImageDraw, ImageFont, ImageOps
from time import time
from tensorflow.keras.models import load_model

def load_frames(video_path):
    'Loads .avi video into array'
    
    frames = []
    v = open(video_path)
    for packet in v.demux():
        for frame in packet.decode():
            img = frame.to_image()
            arr = np.asarray(img)
            frames.append(arr)
    return frames

def equalize_intensity(img):
    p2, p98 = np.percentile(img, (0, 18))
    return rescale_intensity(img, in_range=(p2,p98))

def rgb_threshold(im, thresholds):
    'Thresholds RGB image by given RBG thresholds'
    
    c = im.copy()
    mask = c[:,:,0] > thresholds[0][0]
    for i, (l_thr, u_thr) in enumerate(thresholds):
        mask &= (c[:,:,i] > l_thr)
        mask &= (c[:,:,i] < u_thr)
    c[~mask] = (0,0,0)
    return c

def create_index_grid(image_shape, distance):
    'Creates an evenly spaced grid of coordinates across an image'
               
    return [(x, y) for x in range(0, image_shape[0], distance) for y in range(0, image_shape[1], distance)]

def collect_region(seed, visited, im, lower_threshold, upper_threshold):
    'Returns a region of pixel coordinate neighbours withing thresholds'

    detected = set([seed])
    region = set()

    x_min = y_min = 0
    x_max, y_max = im.shape
    
    while len(detected):
        
        pix = detected.pop()
        
        if pix in visited: continue
                
        x, y = pix
    
        for xi in range(max(x-1, x_min), min(x+2, x_max), 2):
            if ((xi, y)) in visited: continue
            if (lower_threshold < im[xi, y] < upper_threshold): detected.add((xi, y))
        for yi in range(max(y-1, y_min), min(y+2, y_max), 2):
            if ((x, yi)) in visited: continue
            if (lower_threshold < im[x, yi] < upper_threshold): detected.add((x, yi))
                
        region.add(pix)
        visited.add(pix)
        
    return list(region)

def collect_all_regions(seeds, im, min_region_size, max_region_size, l_thr, u_thr):
    'Runs collectRegion for every seed and returns a list of all connex regions in the image'
    
    regions = []
    visited = set()

    for seed in seeds:
        
        if seed in visited: continue
            
        region = collect_region(seed, visited, im, l_thr, u_thr)
        
        if min_region_size <= len(region) <= max_region_size: regions.append(region)
        
    return np.array(regions)

def rgb_to_binary(im, invert=False, threshold=200):
    c = im.copy()
    grayscale = (rgb2gray(c)*255).astype('uint8')
    
    if invert: mask = grayscale > threshold
    else: mask = grayscale < threshold
    
    grayscale[mask] = 0
    grayscale[~mask] = 1
    
    return grayscale

def locate_frame(region):
    max_x = max(region[:,0])
    max_y = max(region[:,1])
    min_x = min(region[:,0])
    min_y = min(region[:,1])
    
    return max_x, max_y, min_x, min_y

def draw_bw_rectangle(im_shape, max_x, max_y, min_x, min_y):
        
    g = np.zeros(im_shape)
    
    g[max_x-3:max_x+3, min_y:max_y] = 255
    g[min_x-3:min_x+3, min_y:max_y] = 255

    g[min_x:max_x, min_y-3:min_y+3] = 255
    g[min_x:max_x, max_y-3:max_y+3] = 255
    
    return g

def gray_to_color(gray_frame, color):
    COLORS = {'red':0, 'green':1, 'blue':2}
    assert (color in COLORS)
    
    c = COLORS[color]
    rgb_cell = [1,1,1]
    rgb_cell[c] = 255
    
    new_shape = (gray_frame.shape[0], gray_frame.shape[1], 3)
    rgb_frame = np.zeros(new_shape).astype('uint8')  
    
    rgb_frame[np.where(gray_frame!=0)] = rgb_cell
    
    return rgb_frame

def locate_rgb_regions(rgb_im, seed, min_size, max_size, l_thr, u_thr):
    c = rgb_im.copy()
    black_white = rgb_to_binary(c, threshold=10)*255
    seeds = create_index_grid(black_white.shape, seed)
    regions = collect_all_regions(seeds, black_white, min_size, max_size, l_thr, u_thr)
    return regions

def overlap_frames(underlying, overlying):
    c = underlying.copy()
    c[np.where(overlying)] = overlying[np.where(overlying)]
    return c

def extract_candidate_frame(im, max_x, max_y, min_x, min_y):
    
    x_limit, y_limit = im.shape[:-1]
    
    width = max(max_x-min_x, max_y-min_y)

    x_delta = width - (max_x-min_x)
    min_x -= x_delta//2
    buffer = min(0, min_x)
    
    max_x += x_delta//2
    'If border point'
    if buffer: 
        max_x += (-buffer)
        min_x = 0
    else:
        buffer = max_x - (x_limit-1)
        if buffer > 0: 
            min_x -= buffer
            max_x = x_limit-1
    
    y_delta = width - (max_y-min_y)
    min_y -= y_delta//2
    buffer = min(0, min_y)

    max_y += y_delta//2
    'If border point'
    if buffer: 
        max_y += (-buffer)
        min_y = 0
    else:
        buffer = max_y - (y_limit-1)
        if buffer > 0: 
            min_y -= buffer
            max_y = y_limit-1
            
    img = im.copy()
    img = img[min_x:max_x, min_y:max_y]
        
    return np.array(img)

def add_colored_thumbnail(original_frame, candidate, candidate_validity):
    
    'Stretch the candidate'
    c = Image.fromarray(candidate).resize((128, 128))
    
    'Make a white or red background depending on candidate validity'
    if candidate_validity: 
        bg = np.array([[[255,255,255] for _ in range(128)] for _ in range(128)]).astype('uint8')
    else: 
        bg = np.array([[[255,0,0] for _ in range(128)] for _ in range(128)]).astype('uint8')
        
    bg = Image.fromarray(bg)
    
    'Blend the candidate image and the colored background'
    c = Image.blend(bg, c, alpha=0.5)
    
    'And apply it to the bottom right corner of the original frame'
    frame = original_frame.copy()
    frame[-128:, -128:] = c
       
    return frame

def crop_content(frame):

    f = frame.copy()
    max_x, max_y = f.shape[:-1]

    x_0, y_0 = (max_x, max_y)

    'Find content'
    for xi in range(max_x):
        for yi in range(max_y):
            if (f[xi, yi] < np.array([225,225,225])).any():
                x_0 = min(xi, x_0)
                y_0 = min(yi, y_0)

    x_1, y_1 = (x_0, y_0)
    for xi in range(max_x-1, x_0, -1):
        for yi in range(max_y-1, y_0, -1):
            if (f[xi, yi] < np.array([225,225,225])).any():
                x_1 = max(xi, x_1)
                y_1 = max(yi, y_1)
                
    return f[x_0:x_1, y_0:y_1]

def pad_quadratic(frame):
    y, x = frame.shape[:-1]
    f = frame.copy()
    mx = max(x, y)
    padded = ImageOps.expand(Image.fromarray(f), ((mx-x)//2, (mx-y)//2), fill='white')
    return np.array(padded)

def periferal_pixels(im, width, threshold=0):

    gray = rgb_to_binary(im, invert=True)
    max_x, max_y = gray.shape

    perifery_pixels = sum(gray[0:width, 0:max_y].ravel()) + sum(gray[max_x-width:max_x, 0:max_y].ravel())
    perifery_pixels += sum(gray[0:max_x, 0:width].ravel()) + sum(gray[0:max_x, max_y-width:max_y].ravel())
    
    return perifery_pixels > threshold 

def calculator():
    equation = ' '
    s = ' '  
    while s != '=':
        s = yield s
        equation += s  
    yield eval(equation[:-1])

def visualize_equation(shape):
    mask = np.zeros(shape).astype('uint8')
    
    max_y, max_x = mask.shape[:-1]
    
    mask[-64:, :-128, :] = [1, 1, 1] 
    
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    
    offset = 5
    font_size = 25
    font = ImageFont.truetype('arial', font_size)
    draw.text((offset, max_y-40),"Equation: ",(255,255,255), font=font)
    
    offset += 0.6*font_size*8
    
    frame, validity, coord, symbol = (None, None, (0, 0), None)
    last_coord = None
    while True:
        f = (yield frame)
        v = (yield validity)
        y,x = (yield coord)
        s = (yield symbol)
        
        new_coord = (x, y)
        if v:
            draw.text((offset, max_y-40), s ,(255,255,255), font=font)
            offset += 0.6*font_size*1.3
        if last_coord:
            draw.line((last_coord, new_coord), width=5, fill=(1,255,1))
        last_coord = new_coord
        yield overlap_frames(f, np.array(mask))

def poke_visualizer(viz, im, coord, symbol, valid, result):
    next(viz)
    viz.send(im), viz.send(valid), viz.send(coord)
    frame = viz.send(symbol)
    if symbol == '=':
        next(viz)
        viz.send(im), viz.send(valid), viz.send(coord)
        frame = viz.send(str(result))
    return frame

def make_video(frames, path):
    container = open(path, mode='w')
    
    stream = container.add_stream('rawvideo', rate=2)
    (h, w) = frames[0].shape[:-1]

    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'
    
    for f in frames:
        frame = VideoFrame.from_ndarray(f, format='rgb24')
        for packet in stream.encode(frame): container.mux(packet)
    for packet in stream.encode(): container.mux(packet)
    container.close()

def classifier():
    mod = load_model('models/mod_3_epochs')
    d = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '+',
        10: '/',
        11: '-',
        12: '*',
        13: '='
    }
    
    image = None
    while True:
        im = (yield image)
        im = Image.fromarray(im).resize((16,16))
        im = np.array(ImageOps.invert(ImageOps.expand(ImageOps.invert(im), (6,6))))
        im = rgb_to_binary(im, invert=True).reshape(1,28,28,1)
        pred = mod.predict_classes(im)[0]
        yield d[pred]


if __name__ == '__main__':

    major_tick = time()

    # Fetch i/o data
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', dest='output')
    parser.add_argument('-i', '--input', dest='input')
    args = parser.parse_args()
    
    src_path = args.input
    output_path = args.output

    assert len(src_path), 'No input path read'
    assert len(output_path), 'No output path read'

    print('(main) Loading frames')
    frames = load_frames(src_path)
    n = len(frames)
    print('(main) Loaded {} frames'.format(n))

    print('(main) Loading classifier')
    clf = classifier()
    next(clf)
    print('(main) Loaded classifier')

    # Feed me. Yields result whed fed '='
    calc = calculator()
    next(calc)

    # Keeps track of the equation visualization state
    viz = visualize_equation(frames[0].shape)

    # Helper variables for equation integrity purposes
    result = 0
    symbols = ' '
    active_equation = True

    # For video output
    output_frames = []

    print('\n Frame \tValid\tClass\tTime\t ')
    print('+-----------------------------+')

    OPERATORS = ['+', '-', '/', '*', '=']

    last_symbol_was_operator = True

    for i, f in enumerate(frames):

        tic = time()
        print('   {}'.format(i+1), end='\t')

        # Do not classify this frame if:
        #    i) '=' has already been classified
        #   ii) Previous frame was classified (car will not manage
        #       to move from one clear shot to another in one frame) 
        valid = active_equation and (symbols[-1] == 'N')

        # Symbol subject to change if a valid classification is made
        symbol = 'N'

        # Equalize intensity and filter the (red) arrow
        eqf = equalize_intensity(f)
        # parameters: image, ((red_thresholds), (green_thresholds), (blue_thresholds))
        arrow = rgb_threshold(eqf, ((180, 256), (-1,190), (-1,190)))

        # Locate arrow indices using region growing
        # parameters: image, seed_grid_spacing, min_region_size, max_region_size, l_pixel_threshold, u_pixel_threshold
        arrow_regions = locate_rgb_regions(arrow, 10, 1000, 3000, 10, 256)
        assert (len(arrow_regions)==1), 'Found no arrow in frame {}'.format(i)

        # Draw surrounding rectangle
        max_x, max_y, min_x, min_y = locate_frame(arrow_regions[0])
        center_coord = (min_x + (max_x-min_x)//2, min_y + (max_y-min_y)//2)
        bw_rectangle = draw_bw_rectangle(arrow.shape[:-1], max_x, max_y, min_x, min_y)
        rgb_rectangle = gray_to_color(bw_rectangle, 'green')

        # Assume all symbols are visible in first frame
        # Use this as a reference for later
        if i == 0: 
            reference_region = set([tuple(a) for a in arrow_regions[0]])
            reference_frame = eqf

        # Extract candidate from reference frame, 
        # corresponding to the area beneath vehicle in this frame
        candidate = extract_candidate_frame(reference_frame, max_x, max_y, min_x, min_y)

        # Discard candidate if there exist objects on the border
        # i.e. we do not have an entirely encapsulated object in the image
        valid &= not periferal_pixels(candidate, 5)
        
        if valid: 
            # Discard candidate if frame mostly white
            valid = sum(candidate.ravel()) / len(candidate.ravel())<254
            if valid: 
                # Discard candidate if it overlaps with the reference frame
                valid = len(set([tuple(a) for a in arrow_regions[0]])&reference_region)==0

                if valid:
                    # Candidate fit for prediction
                    candidate = crop_content(candidate)
                    candidate = pad_quadratic(candidate)
                    prediction = clf.send(candidate)
                    next(clf)

                    # Analyze the result. Every other classification should be an operator
                    valid = False
                    if prediction in OPERATORS:
                        if not last_symbol_was_operator: 
                            symbol = prediction
                            valid = True
                            last_symbol_was_operator = True
                    else:
                        if last_symbol_was_operator: 
                            symbol = prediction
                            valid = True
                            last_symbol_was_operator = False

                    if valid:
                        # Send valid symbol to calculator
                        result = calc.send(symbol)

                        # = terminates the equation and implies no need for further classification
                        active_equation = (symbol != '=')

        
        symbols += symbol
        if i == n-1 and active_equation:
            # Terminate equation if not already terminated at the last frame
            valid = True
            symbol = '='
            active_equation = False
            result = calc.send(symbol)
            
        print('{}'.format(valid), end='\t')
        print('  {}'.format(symbol), end='\t')

        # prepare the output frame
        with_arrow_rect = overlap_frames(f, rgb_rectangle)
        with_thumbnail = add_colored_thumbnail(with_arrow_rect, candidate, valid)
        with_equation = poke_visualizer(viz, with_thumbnail, center_coord, symbol, valid, result)
        output_frames.append(with_equation)
        toc = time()    
        print(f'{toc-tic:.2f}s')
    make_video(output_frames, output_path)
    
    major_tock = time()
    print('+-----------------------------+')
    print(f'   Total time: {major_tock-major_tick:.2f}s')