# Screenshot Q&A Examples

## Test 1: App dashboard

Input: Screenshot of a Grafana dashboard showing CPU metrics.

Question: What metric is being displayed and what is the current value?

### LLaVA 7B
'The image shows a Grafana dashboard displaying CPU usage metrics. The current value appears to be around 73% for the main server.'
Accurate.

### Qwen-VL-Chat
'A monitoring dashboard with CPU metrics. Current CPU usage is approximately 73% with a spike visible around 14:00.'
Accurate, noticed the spike.

## Test 2: Settings page

Input: Screenshot of Android settings page.

Question: What Wi-Fi network is connected?

### LLaVA 7B
'The device is connected to a Wi-Fi network called HomeNetwork.'
Correct.

### Qwen-VL-Chat
'Settings page showing Wi-Fi connected to HomeNetwork with signal strength of 3 bars.'
More detailed.

## Test 3: Error dialog

Input: Screenshot of a Python error traceback in terminal.

Question: What error occurred and on which line?

### LLaVA 7B
'A TypeError occurred on line 42.'
Partially correct -- line was 45.

### Qwen-VL-Chat
'A TypeError: unsupported operand type(s) for + occurred on line 45 of main.py.'
Correct and detailed.

## Observations

- Both models handle clean screenshots well
- Low-resolution screenshots degrade quality significantly
- Text-heavy screenshots (code, logs) work better with Qwen-VL
- UI element detection (buttons, menus) is hit-or-miss
