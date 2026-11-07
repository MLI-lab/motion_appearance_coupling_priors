from functools import partial

def resolve_flow(options, name : str, **args):
    if name == "recon_and_baseline_tuning_with_pretrained_dms_flow":
        from src.flows.tuned_recon_with_pretrained_dms_and_baselines import recon_and_baseline_tuning_with_pretrained_dms_flow
        return partial(recon_and_baseline_tuning_with_pretrained_dms_flow.with_options(**options), **args)
    elif name == "recon_with_pretrained_dms":
        from src.flows.recon_with_pretrained_dm_flow import recon_with_pretrained_dms
        return partial(recon_with_pretrained_dms.with_options(**options), **args)
    else:
        raise ValueError(f"Flow {name} is not defined or not implemented.")