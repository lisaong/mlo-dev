# Start from a base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Get arguments from the docker build command
ARG USER_ID
ARG GROUP_ID

# Create a new user and group with the given IDs
RUN addgroup --gid $GROUP_ID "usergroup"
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN adduser user sudo

# Install sudo and add the user to the sudo group
RUN apt-get update && apt-get install -y sudo
RUN usermod -aG sudo user
RUN echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install dependencies
RUN apt-get update && apt-get install -y build-essential curl git zlib1g-dev

# Switch to the new user
USER user
