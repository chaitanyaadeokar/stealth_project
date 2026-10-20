import sys
import os
import time

# Ensure whisper/ directory is accessible
_whisper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _whisper_dir not in sys.path:
    sys.path.insert(0, _whisper_dir)

from sentence_classifier import SentenceClassifier, classify_sentence, get_classifier

sentence = "what is the name of capital of france"

print("=" * 60)
print(" ModernBERT-base Sentence Classification Benchmark")
print("=" * 60)

# 1. Measure Cold Start (Model load + 1st inference)
print("\n[1] Initializing model & Cold Start...")
t_cold_start = time.perf_counter()
res_cold = classify_sentence(sentence)
cold_ms = (time.perf_counter() - t_cold_start) * 1000.0

print(f"    Cold Start Latency: {cold_ms:.2f} ms")
print(f"    Classification   : {res_cold['classifications']} ({res_cold['confidence']*100:.1f}%)")

# 2. Measure Warm Steady-State Inference (5 runs)
print("\n[2] Benchmarking Steady-State Inference (Warm)...")
classifier = get_classifier()
runs = 5
warm_times = []

for i in range(1, runs + 1):
    t0 = time.perf_counter()
    res = classifier.classify(sentence)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    warm_times.append(dt_ms)
    print(f"    Run {i}: {dt_ms:.2f} ms")

avg_ms = sum(warm_times) / len(warm_times)
min_ms = min(warm_times)
max_ms = max(warm_times)

print("\n" + "-" * 60)
print(f" Sentence : \"{sentence}\"")
print(f" Result   : {res['top_category']} (Scores: {res['scores']})")
print(f" Min Time : {min_ms:.2f} ms")
print(f" Max Time : {max_ms:.2f} ms")
print(f" Avg Time : {avg_ms:.2f} ms")
print("=" * 60)
