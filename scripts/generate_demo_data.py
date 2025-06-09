from pathlib import Path

from defi_security.demo_data import generate_demo_data


def main() -> None:
    destination = Path("data/sample/incidents.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_data().to_csv(destination, index=False)
    print(f"Wrote explicit synthetic demo fixture to {destination}")


if __name__ == "__main__":
    main()

