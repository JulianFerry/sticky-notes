from PIL import Image

def preview(img, size=200):
    """
    Preview an image as a thumbnail
    
    Arguments:
    - img  - numpy array: Image to display
    - size - int:         Max image width/height in pixels (aspect ratio is preserved)
    """
    img = img.copy()
    # Calculate height and width, keeping the aspect ratio
    h = img.shape[0]
    w = img.shape[1]
    if w >= h:
        w = int((w / h) * size)
        h = size
    else:
        h = int((w / h) * size)
        w = size
    # Display image
    img_thumbnail = Image.fromarray(img)
    img_thumbnail.thumbnail((h, w))
    display(img_thumbnail)