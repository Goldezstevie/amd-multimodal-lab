# Image Preprocessing Notes

## For screenshots

Screenshots are usually clean digital images. Minimal preprocessing needed:
1. Resize to model's expected resolution (usually 336x336 or 448x448)
2. Convert to RGB (remove alpha channel if present)
3. Normalize with model's mean/std

## For photos of documents

Phone camera photos need more work:
1. Perspective correction (deskew)
2. Contrast enhancement
3. Binarization (convert to black/white)
4. Noise reduction

## For charts and graphs

Charts are tricky because:
- Small text labels need to be readable
- Colors carry information (legend, data series)
- Axes need to be visible

Best approach: keep original resolution, don't crop too aggressively.

## Pipeline

```python
from PIL import Image, ImageEnhance

def preprocess_screenshot(img_path, target_size=336):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((target_size, target_size))
    return img

def preprocess_scan(img_path, target_size=336):
    img = Image.open(img_path).convert('RGB')
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = img.resize((target_size, target_size))
    return img
```
