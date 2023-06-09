FROM gentoo/stage3

RUN emerge --sync

RUN emerge dev-vcs/git

RUN rmdir /etc/portage/{package.use,package.accept_keywords}

RUN echo -e '[gentoo-repo]\n\
location = /var/db/repos/gentoo-repo\n\
sync-type = git\n\
sync-uri = https://github.com/denisstrizhkin/gentoo-repo.git' > /etc/portage/repos.conf

RUN emerge --sync gentoo-repo

RUN echo -e 'dev-util/nvidia-cuda-toolkit NVIDIA-CUDA\n\
x11-drivers/nvidia-drivers NVIDIA-r2' > /etc/portage/package.license

RUN echo -e 'dev-util/nvidia-cuda-toolkit' > /etc/portage/package.accept_keywords

RUN echo -e 'media-libs/libglvnd X\n\
x11-libs/cairo X' > /etc/portage/package.use

RUN emerge dev-util/nvidia-cuda-toolkit

# echo "dev-util/nvidia-cuda-toolkit NVIDIA-CUDA" > /etc/portage/package.license
# echo dev-util/nvidia-cuda-toolkit > /etc/portage/package.accept_keywords
# echo "x11-drivers/nvidia-drivers NVIDIA-r2" >> /etc/portage/package.accept_keywords