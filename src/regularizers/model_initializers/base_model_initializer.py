from functools import partial

def model_initializer_resolver(name, **params):
    if name == "video_and_motion_init_full":
        from src.regularizers.model_initializers.video_frame_init import video_and_motion_init_full
        return partial(video_and_motion_init_full, **params)
    else:
        raise ValueError(f"Unknown model initializer name: {name}.")
