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
(4096, 1024) | 10.66 | 9.5 | 1.122105263
(4096, 2048) | 15.94 | 12.48 | 1.27724359
(4096, 3072) | 21.92 | 16.19 | 1.353922174
(4096, 4096) | 27.14 | 20.61 | 1.316836487
(4096, 5120) | 34.27 | 24.45 | 1.401635992
(4096, 6144) | 43.81 | 30.43 | 1.439697667
(4096, 7168) | 48.77 | 38.02 | 1.282745923
(4096, 8192) | 55.26 | 42.46 | 1.301460198
(4096, 9216) | 66.66 | 46.02 | 1.448500652
(4096, 10240) | 71.33 | 52.7 | 1.353510436
(4096, 11264) | 74.91 | 70.43 | 1.063609257
(4096, 12288) | 79.36 | 73.95 | 1.073157539
(4096, 13312) | 82.11 | 78.82 | 1.041740675
(4096, 14336) | 85.15 | 81.98 | 1.038667968
(4096, 15360) | 88.45 | 86.05 | 1.027890761
(4096, 16384) | 92.83 | 91.36 | 1.016090193
(4096, 20480) | 117.79 | 116.77 | 1.00873512
(4096, 24576) | 134.11 | 136.67 | 0.98126875
(4096, 28672) | 175.1 | 162.18 | 1.07966457
(4096, 32768) | 224.45 | 181.66 | 1.235549928

The performance gains on the B200 primarily come from 256-bit vectorized loading.

## H20(with CUDA 12.9)
(Batch, Hidden Dim) | FlashInfer(us) | Ours(us) | Speed Up
--- | --- | --- | ---
(4096, 1024) | 14.05 | 10.3 | 1.36407767
(4096, 2048) | 24.93 | 17.86 | 1.395856663
(4096, 3072) | 36.45 | 28.61 | 1.274030059
(4096, 4096) | 48.54 | 38.18 | 1.271346255
(4096, 5120) | 61.66 | 49.44 | 1.247168285
(4096, 6144) | 80.29 | 65.15 | 1.2323868
(4096, 7168) | 92.45 | 73.25 | 1.262116041
(4096, 8192) | 101.57 | 82.5 | 1.231151515
(4096, 9216) | 135.49 | 104.74 | 1.293584113
(4096, 10240) | 142.62 | 113.86 | 1.252590901
(4096, 11264) | 152.26 | 122.08 | 1.247214941
(4096, 12288) | 158.69 | 125.89 | 1.26054492

## H200(with CUDA 12.9)
(Batch, Hidden Dim) | FlashInfer(us) | Ours(us) | Speed Up
--- | --- | --- | ---
(4096, 1024) | 11.39 | 10.62 | 1.072504708
(4096, 2048) | 18.98 | 17.25 | 1.100289855
(4096, 3072) | 26.4 | 24.86 | 1.061946903
(4096, 4096) | 34.08 | 34.21 | 0.996199942
(4096, 5120) | 42.56 | 41.79 | 1.018425461
(4096, 6144) | 53.82 | 49.98 | 1.076830732
(4096, 7168) | 60.67 | 58.59 | 1.035500939
(4096, 8192) | 68.19 | 69.02 | 0.9879745
(4096, 9216) | 88.13 | 78.24 | 1.12640593
(4096, 10240) | 91.55 | 85.89 | 1.065898242
(4096, 11264) | 101.44 | 94.08 | 1.078231293
(4096, 12288) | 103.9 | 102.24 | 1.016236307

There are two main differences between Flashinfer and hp_rms_norm:
1. The Flashinfer implementation includes additional STS (storing float type inp+res to shared memory) and LDS.
2. Includes more integer instructions (ALU), uniform instructions (Uniform Datapath), and utilization of ADU Pipe.

Notice: The H200 operates at a higher frequency (1.97GHz > 1.81GHz).

For H20, these additional operations incur significant overhead (including Reduce computation), leading to a performance bottleneck.

For H200, however, the overhead (percentage) of these operations is smaller, and the performance bottleneck remains concentrated on Global Memory access. Since the amount of data accessed in Global Memory is similar, the performance difference between the Flashinfer implementation and hp_rms_norm is not significant.

Taking the Flashinfer implementations on H20 and H200 as examples, H200's peak HBM bandwidth is 4.8TB/s, while H20's is 4.0TB/s. If the overhead of redundant operations were the same percentage on both H200 and H20, their actual memory bandwidth usage should also be similar. Therefore, H200's bandwidth utilization **rate** should be lower than H20's (they can both utilize the same bandwidth, but H200's peak bandwidth is higher). However, in reality, H200's bandwidth utilization is higher—clearly, H200 has a higher peak HBM bandwidth and a higher utilization rate, indicating that the main performance bottleneck is concentrated on Global Memory access, with redundant operations accounting for a smaller percentage.