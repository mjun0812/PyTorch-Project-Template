import torch
from omegaconf import OmegaConf

from src.models.backbone import BackboneConfig, build_backbone, get_available_backbones


def test_available_backbones() -> None:
    """利用可能なバックボーンモデルのリストをテスト"""
    available_backbones = get_available_backbones()
    assert len(available_backbones) > 0
    print(available_backbones[:10])


def test_torchvision_backbone() -> None:
    """torchvisionモデルの構築をテスト"""
    cfg = BackboneConfig(
        name="torchvision_resnet18",
        pretrained=False,  # テスト高速化のため事前学習なし
        args=OmegaConf.create({"out_indices": [1, 2, 3, 4]}),
    )

    backbone, _ = build_backbone(cfg)

    # モデルが入力を処理できるか確認（小さいサイズの入力を使用）
    dummy_input = torch.randn(1, 3, 224, 224)
    outputs = backbone(dummy_input)
    assert len(outputs) == 4


def test_timm_backbone() -> None:
    """timmモデルの構築をテスト(GPUなしでも実行可能)"""
    cfg = BackboneConfig(
        name="resnet18",
        pretrained=False,  # テスト高速化のため事前学習なし
        args=OmegaConf.create({"out_indices": [1, 2, 3, 4]}),
    )

    backbone, _ = build_backbone(cfg)

    # モデルが入力を処理できるか確認（小さいサイズの入力を使用）
    dummy_input = torch.randn(1, 3, 224, 224)
    outputs = backbone(dummy_input)
    assert len(outputs) == 4


def test_internimage_backbone() -> None:
    """InternImageモデルの構築をテスト"""
    cfg = BackboneConfig(
        name="internimage_t_1k_224",
        pretrained=False,  # テスト高速化のため事前学習なし
        args=OmegaConf.create({"out_indices": [1, 2, 3, 4]}),
    )

    backbone, _ = build_backbone(cfg)

    # モデルが入力を処理できるか確認（小さいサイズの入力を使用）
    dummy_input = torch.randn(1, 3, 224, 224)
    outputs = backbone(dummy_input)
    assert len(outputs) == 4
