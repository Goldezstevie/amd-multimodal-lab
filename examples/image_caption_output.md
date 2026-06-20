# Image Captioning Output

## Test images and captions

### Bar chart
Input: Simple bar chart showing monthly revenue.

LLaVA: 'A bar chart displaying monthly revenue figures with values ranging from 10K to 50K.'
Qwen-VL: 'Bar chart showing revenue by month. January: 10K, February: 25K, March: 50K. Trend is upward.'
Qwen-VL is more detailed.

### Product photo
Input: Photo of a laptop on a desk.

LLaVA: 'A laptop computer sitting on a wooden desk.'
Qwen-VL: 'A silver laptop on a wooden desk with a coffee cup and notebook nearby.'
Both good.

### Handwritten notes
Input: Photo of handwritten meeting notes.

LLaVA: 'Handwritten text on white paper.'
Qwen-VL: 'Handwritten meeting notes mentioning Q3 targets and budget allocation.'
Qwen-VL can read some of the handwriting.

### Screenshot of code
Input: VS Code screenshot with Python code.

LLaVA: 'A code editor showing Python programming code.'
Qwen-VL: 'VS Code editor displaying a Python function that processes CSV data using pandas.'
Qwen-VL understands the code context.

## Memory usage

| Model | VRAM | Speed |
|-------|------|-------|
| LLaVA 7B | 8.2GB | ~3s per image |
| Qwen-VL-Chat | 6.5GB | ~2s per image |
| MiniCPM-V 2.6 | 4.8GB | ~1.5s per image |
