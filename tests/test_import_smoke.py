def test_package_imports():
    from plant_disease import __version__

    assert __version__ == "0.1.0"
