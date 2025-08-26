from pathlib import Path

from defi_security.reporting import validate_readme_metrics


def main() -> None:
    validate_readme_metrics(Path("README.md"), Path("artifacts/portfolio/metrics.json"))
    print("README portfolio metrics match the machine-readable artifact")


if __name__ == "__main__":
    main()

