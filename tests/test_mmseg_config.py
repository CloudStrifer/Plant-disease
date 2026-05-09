from pathlib import Path


def test_improved_mmseg_config_uses_pretrained_binary_setup() -> None:
    config_path = Path("configs/mmseg/segformer_mit-b0_plantseg_512_pretrained_binary.py")
    text = config_path.read_text(encoding="utf-8")

    assert "mit_b0" in text
    assert "out_channels=1" in text
    assert "use_sigmoid=True" in text
    assert "RandomCrop" in text
    assert "RandomResize" in text
