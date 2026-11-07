
def representation_resolver(name: str, **model_params):
    if "videovoxel" in name.lower() or "video_voxel" in name.lower():
        from src.representations.videovoxel_representation import VideoVoxelRepresentation
        return VideoVoxelRepresentation(**model_params)
    else:
        raise ValueError(f"Unknown representation name: {name}.")
