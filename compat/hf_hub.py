import os


def ensure_hf_hub_compat():
    """
    Compatibility shim for older diffusers code paths running with newer
    huggingface_hub versions.
    """
    import huggingface_hub as hub
    import huggingface_hub.constants as constants

    if not hasattr(hub, "_irasim_hf_hub_compat_wrapped"):
        original_hf_hub_download = hub.hf_hub_download

        def hf_hub_download_compat(*args, **kwargs):
            if "use_auth_token" in kwargs:
                if "token" not in kwargs:
                    kwargs["token"] = kwargs["use_auth_token"]
                kwargs.pop("use_auth_token", None)
            kwargs.pop("resume_download", None)
            return original_hf_hub_download(*args, **kwargs)

        hub.hf_hub_download = hf_hub_download_compat

    if not hasattr(constants, "hf_cache_home"):
        constants.hf_cache_home = constants.HUGGINGFACE_HUB_CACHE

    if not hasattr(hub, "cached_download"):
        hub.cached_download = hub.hf_hub_download

    if not hasattr(hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def path_token():
                return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")

            @staticmethod
            def get_token():
                token_path = HfFolder.path_token()
                if os.path.exists(token_path):
                    with open(token_path, "r", encoding="utf-8") as f:
                        token = f.read().strip()
                        return token or None
                return None

        hub.HfFolder = HfFolder

    hub._irasim_hf_hub_compat_wrapped = True
