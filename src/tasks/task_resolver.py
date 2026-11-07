

from prefect import Task

def resolve_task(name : str) -> Task:
    if name == "recon_task":
        from src.tasks.recon_tasks import recon_task
        return recon_task
    elif name == "recon_baseline_task":
        from src.tasks.recon_baseline_task import recon_baseline_task
        return recon_baseline_task
    elif name == "convert_vimeo_to_video_dataset":
        from src.tasks.convert_vimeo_to_video_dataset_task import convert_vimeo_to_video_dataset
        return convert_vimeo_to_video_dataset
    elif name == "convert_cmrx_to_video_dataset":
        from src.tasks.convert_cmrx_to_video_dataset_task import preprocess_cmrx_to_video_dataset
        return preprocess_cmrx_to_video_dataset
    elif name == "convert_reds_to_video_dataset":
        from src.tasks.convert_reds_to_video_dataset_task import convert_reds_to_video_dataset
        return convert_reds_to_video_dataset
    elif name == "download_dataset":
        from src.tasks.download_dataset_task import download_dataset
        return download_dataset
    elif name == "train_probabilistic_coupling_model":
        from src.tasks.train_probabilistic_coupling_model import train_probabilistic_coupling_model
        return train_probabilistic_coupling_model
    else:
        raise ValueError(f"Task {name} not found")