# High Performance RMSNorm
Traditional RMSNorm reads duplicate data multiple times. We use storage on the GPU SM Core to avoid this.

# Run
```
python setup.py install
python3 test/test_hp_rms_norm.py
```
# Performace
## B200(with CUDA 13.1)
(Batch, Hidden Dim) | FlashInfer(us) | Ours(us) | Speed Up
--- | --- | --- | ---
(4096, 1024) | 10.53 | 9.34 | 1.127408994
(4096, 2048) | 16.51 | 12.35 | 1.336842105
(4096, 3072) | 22.02 | 15.23 | 1.445830598
(4096, 4096) | 26.98 | 18.69 | 1.443552702
(4096, 5120) | 34.5 | 23.78 | 1.450798991
(4096, 6144) | 44.1 | 28.8 | 1.53125
(4096, 7168) | 49.98 | 34.69 | 1.440761026
(4096, 8192) | 54.91 | 39.9 | 1.376190476
(4096, 9216) | 67.2 | 46.43 | 1.447340082
(4096, 10240) | 72.29 | 51.42 | 1.405873201
(4096, 11264) | 76 | 55.14 | 1.378309757
(4096, 12288) | 78.88 | 59.78 | 1.319504851
(4096, 13312) | 82.69 | 80.38 | 1.028738492
(4096, 14336)	| 85.86	| 83.39	| 1.029619858
(4096, 15360)	| 88.77	| 86.4	| 1.027430556
(4096, 16384)	| 93.18	| 90.98	| 1.024181139
(4096, 20480)	| 118.56	| 115.74	| 1.024364956
(4096, 24576)	| 133.64	| 137.31	| 0.973272158
(4096, 28672)	| 175.58	| 161.15	| 1.089543903
(4096, 32768)	| 225.41	| 182.34	| 1.236207086

The performance gains on the B200 primarily come from 256-bit vectorized loading.

## H20(with CUDA 12.9)
(Batch, Hidden Dim) | FlashInfer(us) | Ours(us) | Speed Up
--- | --- | --- | ---
(4096, 1024) | 14.05 | 10.08 | 1.393849206
(4096, 2048) | 24.8	| 16.93	| 1.464855286
(4096, 3072) | 36.64	| 26.88	| 1.363095238
(4096, 4096) | 48.16	| 36.58	| 1.31656643
(4096, 5120) | 61.15	| 47.78	| 1.279824194
(4096, 6144) | 80.22	| 63.07	| 1.271920089
(4096, 7168) | 92.86	| 70.56	| 1.316043084
(4096, 8192) | 100.77	| 79.65	| 1.265160075
(4096, 9216) | 134.98	| 104.03	| 1.297510334
(4096,10240 ) | 143.2	| 113.06	| 1.266584115
(4096, 11264) | 151.97	| 121.54	| 1.250370248
(4096, 12288) | 158.91	| 127.39	| 1.247429155
(4096, 13312) | 167.07	| 137.15	| 1.218155304
(4096, 14336) | 168.7	| 144.06	| 1.171039845
(4096, 15360) | 171.42	| 153.09	| 1.11973349
(4096, 16384) | 177.57	| 161.66	| 1.09841643

## H200(with CUDA 12.9)
(Batch, Hidden Dim) | FlashInfer(us) | Ours(us) | Speed Up
--- | --- | --- | ---
(4096, 1024) | 11.26 | 9.38 | 1.200426439
(4096, 2048) | 17.92	| 17.18	| 1.043073341
(4096, 3072) | 27.04	| 24.45	| 1.10593047
(4096, 4096) | 34.21	| 33.09	| 1.033847084
(4096, 5120) | 43.33	| 41.25	| 1.050424242
(4096, 6144) | 54.08	| 48.9	| 1.10593047
(4096, 7168) | 61.47	| 56.54	| 1.087194906
(4096, 8192) | 67.65	| 65.38	| 1.034720098
(4096, 9216) | 84.26	| 78.46	| 1.073923018
(4096,10240 ) | 96	| 85.25	| 1.126099707
(4096, 11264) | 100.8	| 93.44	| 1.078767123
(4096, 12288) | 105.02	| 103.9	| 1.010779596
(4096, 13312) | 112.06	| 108.86	| 1.029395554
(4096, 14336) | 119.23	| 119.49	| 0.997824086
(4096, 15360) | 125.06	| 130.46	| 0.958608002
(4096, 16384) | 133.98	| 135.23	| 0.990756489

There are two main differences between Flashinfer and hp_rms_norm:
1. The Flashinfer implementation includes additional STS (storing float type inp+res to shared memory) and LDS.
2. Includes more integer instructions (ALU), uniform instructions (Uniform Datapath), and utilization of ADU Pipe.

Notice: The H200 operates at a higher frequency (1.97GHz > 1.81GHz).

For H20, these additional operations incur significant overhead (including Reduce computation), leading to a performance bottleneck.

For H200, however, the overhead (percentage) of these operations is smaller, and the performance bottleneck remains concentrated on Global Memory access. Since the amount of data accessed in Global Memory is similar, the performance difference between the Flashinfer implementation and hp_rms_norm is not significant.

Taking the Flashinfer implementations on H20 and H200 as examples, H200's peak HBM bandwidth is 4.8TB/s, while H20's is 4.0TB/s. If the overhead of redundant operations were the same percentage on both H200 and H20, their actual memory bandwidth usage should also be similar. Therefore, H200's bandwidth utilization **rate** should be lower than H20's (they can both utilize the same bandwidth, but H200's peak bandwidth is higher). However, in reality, H200's bandwidth utilization is higher—clearly, H200 has a higher peak HBM bandwidth and a higher utilization rate, indicating that the main performance bottleneck is concentrated on Global Memory access, with redundant operations accounting for a smaller percentage.