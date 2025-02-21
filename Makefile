.ONESHELL: # runs all commands in the same shell (rather than spawning a new shell for each command)
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
.DEFAULT_GOAL := NONE
.PHONY: NONE env env_lock


# --- Globals
ENV_NAME = parnet # TODO: Change!


# --- Commands

# create/update the environment
env: 
	mamba env update -f environment.yml -n $(ENV_NAME)

# lock / export the current environment to a yaml file
env_lock: 
	mamba env export --from-history --no-builds -n $(ENV_NAME) -f environment.lock.yml
