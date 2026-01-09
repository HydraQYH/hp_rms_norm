# High Performance RMSNorm
Traditional RMSNorm reads duplicate data multiple times. We use storage on the GPU SM Core to avoid this.

# Run
```
python setup.py install
python3 test/test_hp_rms_norm.py

```
