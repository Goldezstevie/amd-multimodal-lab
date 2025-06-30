# Multimodal Lab — Sample Input & Expected Output

## Example 1: Image Captioning (English)

**Input:** A photo of a golden retriever playing on a beach at sunset.

```
Image: golden_retriever_beach.jpg (1920×1080, JPEG)
Model: LLaVA-1.5 7B
Prompt: "Describe this image in detail."
```

**Expected output:**
```
A golden retriever is running along a sandy beach during sunset. The dog
appears to be mid-stride with its tongue out, chasing a tennis ball. The
ocean waves are visible in the background, with warm orange and pink hues
reflecting off the water. The dog's fur is slightly wet, suggesting it had
been swimming earlier.
```

## Example 2: Image Captioning (Indonesian)

**Input:** A photo of a busy street market in Jakarta.

```
Image: jakarta_market.jpg (4032×3024, JPEG)
Model: LLaVA-1.5 7B (multilingual mode)
Prompt: "Jelaskan gambar ini dalam bahasa Indonesia."
```

**Expected output:**
```
Sebuah pasar tradisional yang ramai di Jakarta. Pedagang menjual berbagai
buah-buahan dan sayuran segar di bawah tenda plastik berwarna-warni.
Beberapa pembeli terlihat sedang memilih barang dagangan. Suasana pasar
sangat hidup dengan aktivitas jual beli.
```

## Example 3: Visual Question Answering

**Input:** A chart showing quarterly revenue data.

```
Image: revenue_chart.png (1200×800, PNG)
Questions and expected answers:
```

Q: "What was the revenue in Q3?"
A: "$4.2 million"

Q: "Which quarter had the highest growth?"
A: "Q2 had the highest growth at approximately 23% quarter-over-quarter."

Q: "What's the overall trend?"
A: "Revenue shows a consistent upward trend across all four quarters,
with the largest jump between Q1 and Q2."

## Example 4: Document Understanding

**Input:** A scanned medical report.

```
Image: medical_report_scan.jpg (2480×3508, JPEG, 300 DPI scan)
Prompt: "Extract the patient name, date, and key findings."
```

**Expected output:**
```
Patient: John Doe
Date: March 15, 2025
Key Findings:
- Blood pressure: 120/80 mmHg (normal)
- Heart rate: 72 bpm (regular)
- Cholesterol: 195 mg/dL (borderline)
Recommendation: Follow-up in 6 months, reduce saturated fat intake.
```

## Example 5: CLIP Image-Text Similarity

**Input:**
```
Images: [cat.jpg, dog.jpg, car.jpg, sunset.jpg]
Text query: "A furry animal sitting on a couch"
```

**Expected similarity scores:**
```
cat.jpg:    0.87  ← highest match
dog.jpg:    0.82
sunset.jpg: 0.23
car.jpg:    0.11
```

## Example 6: Multi-Image Document Understanding

**Input:**
```
Images: [page1.jpg, page2.jpg, page3.jpg]  (3-page scanned report)
Prompt: "Summarize the key conclusions from this report."
```

**Expected output:**
```
The report concludes that the new energy policy will reduce carbon emissions
by 15% by 2030. Key recommendations include: increasing solar capacity by
40%, implementing carbon pricing at $50/ton, and providing subsidies for
electric vehicle adoption. The estimated cost is $2.3B over 5 years with
an expected ROI of 3.2x by 2035.
```
