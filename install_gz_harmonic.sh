#!/usr/bin/env bash
set -euo pipefail
. /etc/os-release
arch=$(dpkg --print-architecture)
sudo mkdir -p /usr/share/keyrings
curl -fsSL https://packages.osrfoundation.org/gazebo.gpg | sudo tee /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg >/dev/null
printf 'deb [arch=%s signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable %s main\n' "$arch" "$VERSION_CODENAME" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
sudo apt-get update
sudo apt-get install -y gz-harmonic
gz sim --version
