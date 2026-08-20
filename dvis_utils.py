import colorsys

def get_color(index):
    golden_ratio_conjugate = 0.61
    hue = (index * golden_ratio_conjugate) % 1.0
    
    saturation = 0.65 
    lightness = 0.55
    
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (r, g, b, 1)