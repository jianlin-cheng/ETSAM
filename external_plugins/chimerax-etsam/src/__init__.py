"""ChimeraX bundle for ETSAM membrane segmentation of cryo-ET tomograms."""

from chimerax.core.toolshed import BundleAPI

__version__ = "1.0.0"


class _ETSAMAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def initialize(session, bi):
        # Runs at ChimeraX startup, before the user interface (and with it
        # QtWebEngine, which draws the Log and Help panels) is created.
        if session.ui.is_gui:
            from .manager import apply_web_gpu_workaround

            apply_web_gpu_workaround(session)

    @staticmethod
    def finish(session, bi):
        pass

    @staticmethod
    def start_tool(session, bi, ti):
        from .tool import ETSAMTool

        return ETSAMTool(session, ti.name)

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd

        cmd.register_command(ci.name, logger)

    @staticmethod
    def get_class(class_name):
        if class_name == "ETSAMTool":
            from .tool import ETSAMTool

            return ETSAMTool
        return None


bundle_api = _ETSAMAPI()
