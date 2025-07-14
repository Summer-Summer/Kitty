import argparse

# Adding common configs for RoCK-KV models
def update_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--sink_length",        type=int, default=32,       help="Sink length")
    parser.add_argument("--buffer_length",      type=int, default=128,      help="Buffer length")
    parser.add_argument("--group_size",         type=int, default=128,      help="Group size")
    parser.add_argument("--kbits",              type=int, default=2,        help="Number of bits for K Cache")
    parser.add_argument("--vbits",              type=int, default=2,        help="Number of bits for V Cache")
    parser.add_argument("--promote_ratio",      type=float, default=0.0,    help="Keep ratio (fp16) for mixed-precision K cache, default=0.0")
    parser.add_argument("--promote_bit",        type=int, default=4,        help="Promote bit for K cache, default=4")
    parser.add_argument("--channel_selection",  type=int, default=3,        choices=[-1, 0, 1, 2, 3], help="Channel selection method: 0 for Random, 1 for Variance, 2 for Magnitude, 3 for RoPE-aware")
    return parser