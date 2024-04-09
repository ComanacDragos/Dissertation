class YOLOCommonConfig:
    # ANCHORS = [
    #     [
    #         52.61428092700272,
    #         43.932482638888786
    #     ],
    #     [
    #         301.60267215740106,
    #         175.67956760466703
    #     ],
    #     [
    #         140.1720528001553,
    #         104.9496088517907
    #     ]
    # ]

    ANCHORS = [
        [
            37.35277173913043,
            79.33697931276296
        ],
        [
            38.48622620941601,
            25.792543751104823
        ],
        [
            112.58437867647058,
            68.02158088235294
        ]
    ]
    GRID_SIZE = (20, 64) #(6, 20) #(12, 39)
    BACKBONE_OUTPUTS = (
        # "block_7_add",
        "block_5_add",  # -> 20, 64 (s: 8)
        "block_12_add", # -> 10, 32 (s: 16)
        "block_15_add", # -> 5, 16 (s:32)
    )
