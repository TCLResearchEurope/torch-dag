import logging

import timm

from torch_dag_timm_plugin.constants import SUPPORTED_TIMM_VERSION

logger = logging.getLogger(__name__)

try:
    initial_autowrap_timm_modules = [
        timm.layers.Conv2dSame,
        timm.layers.grn.GlobalResponseNorm,
        timm.layers.norm.LayerNorm,
        timm.layers.norm.LayerNorm2d,
        timm.layers.patch_embed.PatchEmbed,
        timm.layers.GroupNorm1,
        timm.layers.pool2d_same.AvgPool2dSame,
        timm.layers.pool2d_same.MaxPool2dSame,
        timm.layers.std_conv.StdConv2d,
        timm.layers.std_conv.StdConv2dSame,
        timm.layers.std_conv.ScaledStdConv2d,
        timm.layers.std_conv.ScaledStdConv2dSame,
        timm.models.beit.Attention,
        timm.models.byobnet.RepVggBlock,
        timm.models.cait.ClassAttn,
        timm.models.cait.TalkingHeadAttn,
        timm.models.coat.ConvPosEnc,
        timm.models.coat.ConvRelPosEnc,
        timm.models.coat.FactorAttnConvRelPosEnc,
        timm.models.crossvit.CrossAttention,
        timm.models.crossvit.PatchEmbed,
        timm.models.edgenext.CrossCovarianceAttn,
        timm.models.edgenext.PositionalEncodingFourier,
        timm.models.efficientformer_v2.Attention2d,
        timm.models.efficientformer_v2.Attention2dDownsample,
        timm.models.efficientformer_v2.LayerScale2d,
        timm.models.gcvit.WindowAttentionGlobal,
        timm.layers.BatchNormAct2d,
        timm.layers.LayerNorm2d,
        timm.layers.Linear,
        timm.layers.DropPath,
        timm.models.metaformer.Attention,
        timm.models.metaformer.LayerNormNoBias,
        timm.models.metaformer.LayerNorm2dNoBias,
        timm.models.metaformer.GroupNorm1NoBias,
        timm.models.swin_transformer_v2_cr.PatchMerging,
        timm.models.swin_transformer_v2_cr.PatchEmbed,
        timm.models.swin_transformer_v2_cr.WindowMultiHeadAttention,
        timm.models.vision_transformer.Attention,
        timm.models.vision_transformer.PatchEmbed,
        timm.models.vision_transformer.LayerScale,
        timm.models.xcit.PositionalEncodingFourier,
        timm.models.xcit.XCA,
        timm.models.swin_transformer_v2.PatchEmbed,
        timm.models.swin_transformer_v2.PatchMerging,
        timm.models.swin_transformer_v2.WindowAttention,
        timm.models.mobilevit.LinearSelfAttention,
        timm.models.mobilevit.MobileVitV2Block,
    ]
    initial_autowrap_timm_functions = [
        timm.models.crossvit.scale_image,
        timm.models.gcvit.window_partition,
        timm.models.gcvit.window_reverse,
        timm.models.mvitv2.reshape_post_pool,
        timm.models.mvitv2.reshape_pre_pool,
        timm.models.mvitv2.cal_rel_pos_type,
        timm.models.swin_transformer_v2_cr.window_partition,
        timm.models.swin_transformer_v2_cr.window_reverse,
        timm.layers.resample_abs_pos_embed_nhwc,
        timm.layers.apply_rot_embed_cat,
        timm.models.swin_transformer_v2.window_reverse,
        timm.models.swin_transformer_v2.window_partition,
    ]
    unprunable_timm_modules = [
        timm.models.efficientformer_v2.Attention2d,
        timm.models.efficientformer_v2.Attention2dDownsample,
        timm.layers.PatchEmbed,
        timm.models.vision_transformer.PatchEmbed,
        timm.models.beit.Attention,
        timm.models.cait.TalkingHeadAttn,
        timm.models.cait.ClassAttn,
        timm.models.coat.FactorAttnConvRelPosEnc,
        timm.models.coat.ConvPosEnc,
        timm.models.coat.ConvRelPosEnc,
        timm.models.crossvit.CrossAttention,
        timm.models.crossvit.PatchEmbed,
        timm.models.edgenext.PositionalEncodingFourier,
        timm.models.edgenext.CrossCovarianceAttn,
        timm.models.gcvit.WindowAttentionGlobal,
        timm.models.swin_transformer_v2_cr.PatchEmbed,
        timm.models.swin_transformer_v2_cr.PatchMerging,
        timm.models.swin_transformer_v2_cr.WindowMultiHeadAttention,
        timm.models.vision_transformer.PatchEmbed,
        timm.models.xcit.XCA,
        timm.models.xcit.PositionalEncodingFourier,
        timm.models.swin_transformer_v2.PatchEmbed,
        timm.models.swin_transformer_v2.PatchMerging,
        timm.models.swin_transformer_v2.WindowAttention,
        timm.models.mobilevit.LinearSelfAttention,
        timm.models.mobilevit.MobileVitV2Block,
    ]
except AttributeError as e:
    logger.error(
        f"{e}\n"
        f"May originate from unsupported timm version: {timm.__version__}. Supported version is {SUPPORTED_TIMM_VERSION}."
    )
    initial_autowrap_timm_modules = []
    initial_autowrap_timm_functions = []
    unprunable_timm_modules = []
