"""Shared Hydra lifetime for SAM and GEN3C model configuration."""

from contextlib import contextmanager

from hydra import initialize, initialize_config_module
from hydra.core.global_hydra import GlobalHydra


@contextmanager
def model_config_session(*, sam_config: str | None = None):
    """Reuse caller-owned Hydra, or own a session until this context exits.

    Wrap application startup/model loading with sam_config set to the requested
    SAM configuration. Its package becomes the file-config source; GEN3C's
    ConfigStore registrations remain available alongside it. Initialize before
    importing SAM so its package initializer does not take global ownership.

    Existing sessions (including those created by import sam2) are never cleared
    or replaced. SAM callers require their config to be discoverable there.
    Standalone GEN3C callers need only ConfigStore and do not import SAM.
    Configuration loading/registration must be serialized within a process.
    """
    global_hydra = GlobalHydra.instance()
    if global_hydra.is_initialized():
        if sam_config is not None and not any(
            source.is_config(sam_config) for source in global_hydra.config_loader().get_sources()
        ):
            raise RuntimeError(
                f"The active Hydra session cannot resolve SAM configuration {sam_config!r}. "
                "Expose pkg://sam2 at application startup or use model_config_session(sam_config=...) "
                "before either model is loaded. The caller's session was not changed."
            )
        yield
        return

    # initialize_config_module sets up Hydra before SAM is imported/composed.
    # Context ownership ensures cleanup on success and on construction failures.
    session = (
        initialize_config_module("sam2", version_base="1.2")
        if sam_config is not None
        else initialize(version_base=None)
    )
    with session:
        yield
