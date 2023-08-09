auto_scale_lr = dict(base_batch_size=48, enable=False)
backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
data_root = 'data/kitti/'
dataset_type = 'KittiDataset'
db_sampler = dict(
    backend_args=None,
    classes=[
        'Pedestrian',
        'Cyclist',
        'Car',
    ],
    data_root='data/kitti/',
    info_path='data/kitti/kitti_dbinfos_train.pkl',
    points_loader=dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    prepare=dict(
        filter_by_difficulty=[
            -1,
        ],
        filter_by_min_points=dict(Car=5, Cyclist=5, Pedestrian=5)),
    rate=1.0,
    sample_groups=dict(Car=15, Cyclist=10, Pedestrian=10))
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_dataloader = dict(
    dataset=dict(
        metainfo=dict(CLASSES=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -40,
                            -3,
                            70.4,
                            40,
                            1,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ]))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=False, use_lidar=True)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.001
metainfo = dict(
    CLASSES=[
        'Pedestrian',
        'Cyclist',
        'Car',
    ],
    classes=[
        'Pedestrian',
        'Cyclist',
        'Car',
    ])
model = dict(
    backbone=dict(
        in_channels=256,
        layer_nums=[
            5,
            5,
        ],
        layer_strides=[
            1,
            2,
        ],
        out_channels=[
            128,
            256,
        ],
        type='SECOND'),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            max_voxels=(
                16000,
                40000,
            ),
            point_cloud_range=[
                0,
                -40,
                -3,
                70.4,
                40,
                1,
            ],
            voxel_size=[
                0.05,
                0.05,
                0.1,
            ])),
    middle_encoder=dict(
        encoder_paddings=(
            (
                0,
                0,
                0,
            ),
            (
                (
                    1,
                    1,
                    1,
                ),
                0,
                0,
            ),
            (
                (
                    1,
                    1,
                    1,
                ),
                0,
                0,
            ),
            (
                (
                    0,
                    1,
                    1,
                ),
                0,
                0,
            ),
        ),
        in_channels=4,
        order=(
            'conv',
            'norm',
            'act',
        ),
        return_middle_feats=True,
        sparse_shape=[
            41,
            1600,
            1408,
        ],
        type='SparseEncoder'),
    neck=dict(
        in_channels=[
            128,
            256,
        ],
        out_channels=[
            256,
            256,
        ],
        type='SECONDFPN',
        upsample_strides=[
            1,
            2,
        ]),
    points_encoder=dict(
        bev_feat_channel=256,
        bev_scale_factor=8,
        fused_out_channel=128,
        num_keypoints=2048,
        point_cloud_range=[
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ],
        rawpoints_sa_cfgs=dict(
            in_channels=1,
            mlp_channels=(
                (
                    16,
                    16,
                ),
                (
                    16,
                    16,
                ),
            ),
            radius=(
                0.4,
                0.8,
            ),
            sample_nums=(
                16,
                16,
            ),
            type='StackedSAModuleMSG',
            use_xyz=True),
        type='VoxelSetAbstraction',
        voxel_sa_cfgs_list=[
            dict(
                in_channels=16,
                mlp_channels=(
                    (
                        16,
                        16,
                    ),
                    (
                        16,
                        16,
                    ),
                ),
                radius=(
                    0.4,
                    0.8,
                ),
                sample_nums=(
                    16,
                    16,
                ),
                scale_factor=1,
                type='StackedSAModuleMSG',
                use_xyz=True),
            dict(
                in_channels=32,
                mlp_channels=(
                    (
                        32,
                        32,
                    ),
                    (
                        32,
                        32,
                    ),
                ),
                radius=(
                    0.8,
                    1.2,
                ),
                sample_nums=(
                    16,
                    32,
                ),
                scale_factor=2,
                type='StackedSAModuleMSG',
                use_xyz=True),
            dict(
                in_channels=64,
                mlp_channels=(
                    (
                        64,
                        64,
                    ),
                    (
                        64,
                        64,
                    ),
                ),
                radius=(
                    1.2,
                    2.4,
                ),
                sample_nums=(
                    16,
                    32,
                ),
                scale_factor=4,
                type='StackedSAModuleMSG',
                use_xyz=True),
            dict(
                in_channels=64,
                mlp_channels=(
                    (
                        64,
                        64,
                    ),
                    (
                        64,
                        64,
                    ),
                ),
                radius=(
                    2.4,
                    4.8,
                ),
                sample_nums=(
                    16,
                    32,
                ),
                scale_factor=8,
                type='StackedSAModuleMSG',
                use_xyz=True),
        ],
        voxel_size=[
            0.05,
            0.05,
            0.1,
        ]),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            class_agnostic=True,
            cls_channels=(
                256,
                256,
            ),
            dropout_ratio=0.3,
            grid_size=6,
            in_channels=128,
            loss_bbox=dict(
                beta=0.1111111111111111,
                loss_weight=1.0,
                reduction='sum',
                type='mmdet.SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.0,
                reduction='sum',
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True),
            num_classes=3,
            reg_channels=(
                256,
                256,
            ),
            shared_fc_channels=(
                256,
                256,
            ),
            type='PVRCNNBBoxHead',
            with_corner_loss=True),
        bbox_roi_extractor=dict(
            grid_size=6,
            roi_layer=dict(
                in_channels=128,
                mlp_channels=(
                    (
                        64,
                        64,
                    ),
                    (
                        64,
                        64,
                    ),
                ),
                pool_mod='max',
                radius=(
                    0.8,
                    1.6,
                ),
                sample_nums=(
                    16,
                    16,
                ),
                type='StackedSAModuleMSG',
                use_xyz=True),
            type='Batch3DRoIGridExtractor'),
        num_classes=3,
        semantic_head=dict(
            extra_width=0.1,
            in_channels=640,
            loss_seg=dict(
                activated=True,
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                reduction='sum',
                type='mmdet.FocalLoss',
                use_sigmoid=True),
            type='ForegroundSegmentationHead'),
        type='PVRCNNRoiHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ranges=[
                [
                    0,
                    -40.0,
                    -0.6,
                    70.4,
                    40.0,
                    -0.6,
                ],
                [
                    0,
                    -40.0,
                    -0.6,
                    70.4,
                    40.0,
                    -0.6,
                ],
                [
                    0,
                    -40.0,
                    -1.78,
                    70.4,
                    40.0,
                    -1.78,
                ],
            ],
            reshape_out=False,
            rotations=[
                0,
                1.57,
            ],
            sizes=[
                [
                    0.8,
                    0.6,
                    1.73,
                ],
                [
                    1.76,
                    0.6,
                    1.73,
                ],
                [
                    3.9,
                    1.6,
                    1.56,
                ],
            ],
            type='Anchor3DRangeGenerator'),
        assign_per_class=True,
        assigner_per_size=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        diff_rad_by_sin=True,
        dir_offset=0.78539,
        feat_channels=512,
        in_channels=512,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=2.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_dir=dict(
            loss_weight=0.2, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        num_classes=3,
        type='PartA2RPNHead',
        use_direction_classifier=True),
    test_cfg=dict(
        rcnn=dict(
            nms_thr=0.1,
            score_thr=0.1,
            use_raw_score=True,
            use_rotate_nms=True),
        rpn=dict(
            max_num=100,
            nms_post=100,
            nms_pre=1024,
            nms_thr=0.7,
            score_thr=0,
            use_rotate_nms=True)),
    train_cfg=dict(
        rcnn=dict(
            assigner=[
                dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(
                        coordinate='lidar', type='BboxOverlaps3D'),
                    min_pos_iou=0.55,
                    neg_iou_thr=0.55,
                    pos_iou_thr=0.55,
                    type='Max3DIoUAssigner'),
                dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(
                        coordinate='lidar', type='BboxOverlaps3D'),
                    min_pos_iou=0.55,
                    neg_iou_thr=0.55,
                    pos_iou_thr=0.55,
                    type='Max3DIoUAssigner'),
                dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(
                        coordinate='lidar', type='BboxOverlaps3D'),
                    min_pos_iou=0.55,
                    neg_iou_thr=0.55,
                    pos_iou_thr=0.55,
                    type='Max3DIoUAssigner'),
            ],
            cls_neg_thr=0.25,
            cls_pos_thr=0.75,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_iou_piece_thrs=[
                    0.55,
                    0.1,
                ],
                neg_piece_fractions=[
                    0.8,
                    0.2,
                ],
                neg_pos_ub=-1,
                num=128,
                pos_fraction=0.5,
                return_iou=True,
                type='IoUNegPiecewiseSampler')),
        rpn=dict(
            allowed_border=0,
            assigner=[
                dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    min_pos_iou=0.35,
                    neg_iou_thr=0.35,
                    pos_iou_thr=0.5,
                    type='Max3DIoUAssigner'),
                dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    min_pos_iou=0.35,
                    neg_iou_thr=0.35,
                    pos_iou_thr=0.5,
                    type='Max3DIoUAssigner'),
                dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    min_pos_iou=0.45,
                    neg_iou_thr=0.45,
                    pos_iou_thr=0.6,
                    type='Max3DIoUAssigner'),
            ],
            debug=False,
            pos_weight=-1),
        rpn_proposal=dict(
            max_num=512,
            nms_post=512,
            nms_pre=9000,
            nms_thr=0.8,
            score_thr=0,
            use_rotate_nms=True)),
    type='PointVoxelRCNN',
    voxel_encoder=dict(type='HardSimpleVFE'))
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        betas=(
            0.95,
            0.99,
        ), lr=0.001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=15,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=15,
        eta_min=0.01,
        type='CosineAnnealingLR'),
    dict(
        T_max=25,
        begin=15,
        by_epoch=True,
        convert_to_iter_based=True,
        end=40,
        eta_min=1.0000000000000001e-07,
        type='CosineAnnealingLR'),
    dict(
        T_max=15,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=15,
        eta_min=0.8947368421052632,
        type='CosineAnnealingMomentum'),
    dict(
        T_max=25,
        begin=15,
        by_epoch=True,
        convert_to_iter_based=True,
        end=40,
        eta_min=1,
        type='CosineAnnealingMomentum'),
]
point_cloud_range = [
    0,
    -40,
    -3,
    70.4,
    40,
    1,
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        metainfo=dict(
            CLASSES=[
                'Pedestrian',
                'Cyclist',
                'Car',
            ],
            classes=[
                'Pedestrian',
                'Cyclist',
                'Car',
            ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -40,
                            -3,
                            70.4,
                            40,
                            1,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    0,
                    -40,
                    -3,
                    70.4,
                    40,
                    1,
                ],
                type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(pts='training/velodyne_reduced'),
            data_root='data/kitti/',
            metainfo=dict(
                CLASSES=[
                    'Pedestrian',
                    'Cyclist',
                    'Car',
                ],
                classes=[
                    'Pedestrian',
                    'Cyclist',
                    'Car',
                ]),
            modality=dict(use_camera=False, use_lidar=True),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='LIDAR',
                    load_dim=4,
                    type='LoadPointsFromFile',
                    use_dim=4),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    db_sampler=dict(
                        backend_args=None,
                        classes=[
                            'Pedestrian',
                            'Cyclist',
                            'Car',
                        ],
                        data_root='data/kitti/',
                        info_path='data/kitti/kitti_dbinfos_train.pkl',
                        points_loader=dict(
                            backend_args=None,
                            coord_type='LIDAR',
                            load_dim=4,
                            type='LoadPointsFromFile',
                            use_dim=4),
                        prepare=dict(
                            filter_by_difficulty=[
                                -1,
                            ],
                            filter_by_min_points=dict(
                                Car=5, Cyclist=5, Pedestrian=5)),
                        rate=1.0,
                        sample_groups=dict(Car=15, Cyclist=10, Pedestrian=10)),
                    type='ObjectSample',
                    use_ground_plane=True),
                dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
                dict(
                    rot_range=[
                        -0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        0.95,
                        1.05,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    point_cloud_range=[
                        0,
                        -40,
                        -3,
                        70.4,
                        40,
                        1,
                    ],
                    type='PointsRangeFilter'),
                dict(
                    point_cloud_range=[
                        0,
                        -40,
                        -3,
                        70.4,
                        40,
                        1,
                    ],
                    type='ObjectRangeFilter'),
                dict(type='PointShuffle'),
                dict(
                    keys=[
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                    ],
                    type='Pack3DDetInputs'),
            ],
            test_mode=False,
            type='KittiDataset'),
        times=2,
        type='RepeatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        db_sampler=dict(
            backend_args=None,
            classes=[
                'Pedestrian',
                'Cyclist',
                'Car',
            ],
            data_root='data/kitti/',
            info_path='data/kitti/kitti_dbinfos_train.pkl',
            points_loader=dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            prepare=dict(
                filter_by_difficulty=[
                    -1,
                ],
                filter_by_min_points=dict(Car=5, Cyclist=5, Pedestrian=5)),
            rate=1.0,
            sample_groups=dict(Car=15, Cyclist=10, Pedestrian=10)),
        type='ObjectSample',
        use_ground_plane=True),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        rot_range=[
            -0.78539816,
            0.78539816,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        point_cloud_range=[
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ],
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -40,
                            -3,
                            70.4,
                            40,
                            1,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.05,
    0.05,
    0.1,
]
