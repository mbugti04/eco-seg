from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from .main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samat",
        description="SAM Annotation Tool",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to a TOML config file with [paths].data",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to the dataset directory that contains images and classes.json",
    )
    return parser


def _load_config(config_path: Path) -> dict:
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file)


def _resolve_dataset_path(config_path: Path, data_arg: Path | None) -> Path:
    if data_arg is not None:
        return data_arg.expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Provide --data or --config."
        )

    config = _load_config(config_path)
    try:
        data_value = config["paths"]["data"]
    except KeyError as exc:
        raise KeyError(
            f"Missing paths.data in config file: {config_path}"
        ) from exc

    dataset_path = Path(data_value).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = config_path.parent / dataset_path
    return dataset_path.resolve()


def _validate_dataset(dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    images_path = dataset_path / "images"
    if not images_path.is_dir():
        raise FileNotFoundError(
            f"Dataset is missing required images directory: {images_path}"
        )

    classes_path = dataset_path / "classes.json"
    if not classes_path.is_file():
        raise FileNotFoundError(
            f"Dataset is missing required classes file: {classes_path}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    parser = build_parser()
    args, qt_args = parser.parse_known_args(argv[1:])

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve()

    try:
        dataset_path = _resolve_dataset_path(config_path, args.data)
        _validate_dataset(dataset_path)
    except (FileNotFoundError, KeyError, OSError) as exc:
        parser.error(str(exc))

    app = QApplication([argv[0], *qt_args])
    main_window = MainWindow(str(dataset_path))
    main_window.show()
    main_window.load_latest_sample()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())