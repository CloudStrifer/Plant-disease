import yaml


def main():
    with open("configs/baseline_segformer_b0.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    print(f"Loaded baseline config: {config['experiment_name']}")


if __name__ == "__main__":
    main()
