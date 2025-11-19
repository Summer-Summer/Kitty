Benchmarking

--cache_implementation
(0) for KChanBoost;
(1) for FP16 static cache;
(2) for FP16 dynamic cache;
(3) for INT4 quantized cache with HQQ backend."
```
python benchmark_kchanboost.py --max_seq_len 8192 --batch_size 32 --warmup_runs 1 --repeat_runs 1 --cache_implementation 3
python benchmark_kchanboost.py --max_seq_len 8192 --batch_size 8 --warmup_runs 1 --repeat_runs 1 --cache_implementation 2
```

Profiling
```
python benchmark_kchanboost.py --warmup_runs 1 --repeat_runs 0 --cache_implementation
nsys profile -o hf_bs32_maxlen_2048         python benchmark_kchanboost.py --warmup_runs 0 --repeat_runs 1 --cache_implementation
```

```
python benchmark_kchanboost.py --warmup_runs 0 --repeat_runs 1 --cache_implementation
nsys profile -o kchanboost_bs32_maxlen_2048 python benchmark_kchanboost.py --warmup_runs 0 --repeat_runs 1 --cache_implementation
```