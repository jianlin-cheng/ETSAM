"""ChimeraX commands: etsam segment | save | cancel."""

from __future__ import annotations

from chimerax.core.commands import BoolArg, CmdDesc, EnumOf, FloatArg, ModelArg, SaveFileNameArg
from chimerax.core.errors import UserError
from chimerax.map import MapsArg

from .manager import NORMALIZATIONS, STAGE1_PROMPTS, STAGE2_PROMPTS, get_manager


class NotSegmentable(UserError):
    """The volume is not a 3D tomogram."""


def segment_command(volume, options) -> str:
    """The "etsam segment" command for ``volume`` with all of ``options`` spelled out."""
    return (
        f"etsam segment #{volume.id_string} threshold {options['threshold']:.4g}"
        f" split {str(options['split_processing']).lower()}"
        f" cleanup {str(options['cleanup']).lower()}"
        f" normalization {options['normalize_method']}"
        f" stage1Prompt {options['stage1_prompt']} stage2Prompt {options['stage2_prompt']}"
    )


def etsam_segment(
    session,
    maps,
    threshold=None,
    split=None,
    cleanup=None,
    normalization=None,
    stage1_prompt=None,
    stage2_prompt=None,
):
    """
    Segment membranes in open volumes. Each result (the membrane logits, shown as a
    surface at the threshold level) is added as a new volume; nothing is saved (use ChimeraX's "save" command for that). Options that are not given
    use the ETSAM panel's current settings.
    """
    manager = get_manager(session)
    options = manager.options()
    for key, value in (
        ("threshold", threshold),
        ("split_processing", split),
        ("cleanup", cleanup),
        ("normalize_method", normalization),
        ("stage1_prompt", stage1_prompt),
        ("stage2_prompt", stage2_prompt),
    ):
        if value is not None:
            options[key] = value
    if not maps:
        raise UserError("No volumes specified. Open a tomogram first, e.g. with: open file.mrc")
    if session.ui.is_gui and len(maps) > 1:
        raise UserError("Segment one volume at a time (ETSAM runs in the background).")
    for volume in maps:
        try:
            # Without the GUI, wait for each run, so that scripts can continue with the result.
            manager.segment(volume, options, blocking=not session.ui.is_gui)
        except ValueError as e:
            raise NotSegmentable(str(e)) from e
        except RuntimeError as e:
            raise UserError(str(e)) from e


segment_desc = CmdDesc(
    required=[("maps", MapsArg)],
    keyword=[
        ("threshold", FloatArg),
        ("split", BoolArg),
        ("cleanup", BoolArg),
        ("normalization", EnumOf(NORMALIZATIONS)),
        ("stage1_prompt", EnumOf(STAGE1_PROMPTS)),
        ("stage2_prompt", EnumOf(STAGE2_PROMPTS)),
    ],
    synopsis="segment membranes in cryo-ET tomograms with ETSAM",
)


def etsam_save(session, model, path, type="mask"):
    """
    Save an ETSAM result as MRC: the mask at its current threshold (surface level),
    or the logits. ``model`` is the result volume or the segmented tomogram.
    """
    manager = get_manager(session)
    result = next(
        (r for r in manager.results.values() if model in (r.model, r.source) and not r.model.deleted),
        None,
    )
    if result is None:
        raise UserError(f"#{model.id_string} is not an ETSAM result; segment it with 'etsam segment' first.")
    try:
        return manager.save(result, type, path)
    except OSError as e:
        raise UserError(f"Could not save {path}: {e}") from e


save_desc = CmdDesc(
    required=[("model", ModelArg), ("path", SaveFileNameArg)],
    keyword=[("type", EnumOf(["mask", "logits"]))],
    synopsis="save an ETSAM result: the mask at the current threshold, or the logits",
)


def etsam_cancel(session):
    get_manager(session).cancel()


cancel_desc = CmdDesc(synopsis="cancel the running ETSAM segmentation")


def register_command(command_name, logger):
    from chimerax.core.commands import register

    table = {
        "etsam segment": (etsam_segment, segment_desc),
        "etsam save": (etsam_save, save_desc),
        "etsam cancel": (etsam_cancel, cancel_desc),
    }
    func, desc = table[command_name]
    register(command_name, desc, func, logger=logger)
