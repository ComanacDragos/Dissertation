class FCOSCommonConfig:
    STRIDES_WEIGHTS = {
        8: 1.,
        16: 1.,
        # 32: 1.
    }

    THRESHOLDS = [128]

    STRIDES = sorted(STRIDES_WEIGHTS.keys())

    BACKBONE_OUTPUTS = (
        # "block_7_add",
        "block_5_add",   # -> 20, 64 (s: 8)
        "block_12_add",  # -> 10, 32 (s: 16)
        # "block_15_add",  # -> 5, 16 (s:32)
    )
