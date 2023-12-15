import os

def _disable_tensorflow_logs():
    # Disable tensorflow INFO and WARNING log messages. This needs to be 
    # done *before* tensorflow is imported. 
    # TODO: Currently, we are also ignoring erros (levle=3) rather than 
    # just warning (level=2). This could be problematic in the future. 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # %%
    # Disable absl INFO and WARNING log messages
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)

# %%
def _set_tf_dynamic_memory_growth():
    import tensorflow as tf

    # Make sure that tensorflow doesn't gobble up all GPU memory
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass