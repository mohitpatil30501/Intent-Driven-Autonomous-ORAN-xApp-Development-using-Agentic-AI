"""
flexric_rag/utils/flexric_tags.py

FlexRIC-specific file classification helpers.
Maps file paths to (layer, sm_type, is_xapp_example) based on the
known directory structure of the FlexRIC repository.
"""

from pathlib import Path
from typing import Tuple


# ─── Directory → layer map ────────────────────────────────────────────────────

_LAYER_MAP = [
    ("examples/xapp",    "xapp"),
    ("examples/",        "xapp"),          # any example
    ("src/sm/kpm_sm",    "sm"),
    ("src/sm/rc_sm",     "sm"),
    ("src/sm/mac_sm",    "sm"),
    ("src/sm/rlc_sm",    "sm"),
    ("src/sm/",          "sm"),
    ("src/ric/",         "ric"),
    ("src/near-rt-ric",  "ric"),
    ("src/e2ap",         "e2ap"),
    ("src/lib",          "util"),
    ("src/util",         "util"),
    ("include/",         "api"),
]

_SM_MAP = [
    ("kpm",  "E2SM_KPM"),
    ("rc_",  "E2SM_RC"),
    ("/rc/", "E2SM_RC"),
    ("mac",  "E2SM_MAC"),
    ("rlc",  "E2SM_RLC"),
    ("e2ap", "E2AP"),
]


# flexric_tags.py
XAPP_API_FUNCTIONS = {
    # Lifecycle
    "init_xapp_api",
    "try_stop_xapp_api",
    "xapp_wait_end_api",

    # Node discovery
    "e2_nodes_xapp_api",

    # Subscription (report) API
    "report_sm_xapp_api",
    "rm_report_sm_xapp_api",

    # Control API
    "control_sm_xapp_api",

    # Internal helpers (also cross-cutting — called regardless of SM type)
    "valid_global_e2_node",
    "valid_sm_id",
    "static_start_xapp",
    "xapp_unblock_wait_api",
    "sig_handler",
}


def classify_file(rel_path: str) -> Tuple[str, str, bool]:
    """
    Returns (layer, sm_type, is_xapp_example) for a given repo-relative path.
    """
    if chunk["name"] in XAPP_API_FUNCTIONS:
        chunk["sm_type"] = "none"
        chunk["layer"]   = "api"
        chunk["is_xapp_example"] = False
        return chunk


    p = rel_path.lower().replace("\\", "/")

    # layer
    layer = "other"
    for prefix, lyr in _LAYER_MAP:
        if prefix in p:
            layer = lyr
            break

    # sm_type
    sm_type = "none"
    for token, smt in _SM_MAP:
        if token in p:
            sm_type = smt
            break

    is_xapp = "examples" in p

    return layer, sm_type, is_xapp


# ─── Critical boundary function sets ─────────────────────────────────────────

XAPP_E2AP_BOUNDARY = frozenset({
    "e2ap_subscribe",
    "e2ap_unsubscribe",
    "e2ap_control",
    "e2ap_indication_cb",
    "e2ap_setup",
    "e2ap_reset",
})

SM_ENCODE_FUNCS = frozenset({
    "kpm_enc_action_def",
    "kpm_enc_sub_req",
    "kpm_dec_ind_msg",
    "kpm_dec_ind_hdr",
    "rc_enc_ctrl_req",
    "rc_dec_ctrl_resp",
    "mac_enc_action_def",
    "mac_dec_ind_msg",
    "rlc_enc_action_def",
    "rlc_dec_ind_msg",
})

RIC_CORE_FUNCS = frozenset({
    "ric_subscription_request",
    "ric_subscription_response",
    "ric_subscription_delete",
    "ric_indication",
    "ric_control_request",
})
