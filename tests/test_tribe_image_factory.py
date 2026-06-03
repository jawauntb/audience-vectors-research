from __future__ import annotations

import pytest

pytest.importorskip("modal")

from audience_vectors.modal_app import image_factory


def test_tribe_image_pins_exca_with_tribev2_install() -> None:
    requirements = image_factory.TRIBE_UV_PIP_INSTALL_REQUIREMENTS

    assert (
        f"git+https://github.com/facebookresearch/tribev2.git@"
        f"{image_factory.TRIBE_GIT_REF}"
    ) in requirements
    assert "exca==0.5.25" in requirements


def test_tribe_image_import_preflight_covers_exca_and_tribe() -> None:
    command = image_factory._TRIBE_IMPORT_RUNTIME_PREFLIGHT_COMMAND

    assert "import exca.steps.base as exca_base" in command
    assert "exca_base.NoValue()" in command
    assert "from tribev2 import TribeModel" in command
