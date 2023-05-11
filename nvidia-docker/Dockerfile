# Start from a base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Get arguments from the docker build command
ARG USER_ID
ARG GROUP_ID

# Create a new user and group with the given IDs
RUN addgroup --gid $GROUP_ID "usergroup"
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID SOUTHPACIFIC.onglisa

# Switch to the new user
USER user
