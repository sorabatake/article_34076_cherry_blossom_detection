model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[1, 1, 1, 1],
        std=[1, 1 ,1, 1],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        stem_channels=4,
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=256,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss', loss_weight=2.0, eps=5e-6, reduction='mean')),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5
        ),
)

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

img_scales = [(640, 640),]
tta_pipeline = [
    dict(type='LoadTifFromFile', 
         backend_args=None,
         to_float32=True,
         ),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]

train_pipeline = [
    dict(type='LoadTifFromFile', 
         to_float32=True,
         ),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    # dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    # dict(
    #     type='RandomResize',
    #     scale=(640, 640),
    #     ratio_range=(0.1, 2.0),
    #     keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_size=(640, 640),
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    # dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114, 114))),
    # dict(
    #     type='CachedMixUp',
    #     img_scale=(640, 640),
    #     ratio_range=(1.0, 1.0),
    #     max_cached_images=20,
    #     pad_val=(114, 114, 114, 114)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadTifFromFile', 
         backend_args=None,
         to_float32=True,
         ),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114, 114))),
    dict(type='LoadAnnotations', 
         with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataset settings
dataset_type = 'CocoSakuraDataset'
data_root = '/workspace/sample/train_scale1_h640_w640_oh0.0_ow0.0_min1_segs'

backend_args = None

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/_sahi_coco.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/_sahi_coco.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/_sahi_coco.json',
    metric=['bbox', 'segm',],
    format_only=False,
    proposal_nums=(100, 1, 10),
    backend_args=backend_args)
test_evaluator = val_evaluator

# hooks
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
    # dict(
    #     type='PipelineSwitchHook',
    #     switch_epoch=30,
    #     switch_pipeline=train_pipeline_stage2)
]

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=20),
    dict(
        type='MultiStepLR',
        begin=0,
        end=150,
        by_epoch=True,
        milestones=[40, 45,],
        gamma=0.2)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
auto_scale_lr = dict(enable=False, base_batch_size=16)


default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = 'weights/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'
resume = False