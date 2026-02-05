FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG EGSNRC_VERSION=v2025a
ARG EGSNRC_REPO=https://github.com/nrc-cnrc/EGSnrc
ARG EGSNRC_TARBALL_NAME=EGSnrc-${EGSNRC_VERSION}.tar.gz

# Base deps + git/curl for fetching release
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran g++ gcc make \
    libx11-dev libxext-dev libxmu-dev libxi-dev \
    tcl tk expect git curl ca-certificates \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Optional local tarball: place EGSnrc-<version>.tar.gz under egsnrc_edk/sources before building.
COPY egsnrc_edk/sources /tmp/egsnrc_sources

# Fetch or use local EGSnrc archive
WORKDIR /opt
RUN set -eux; \
    if [ -s "/tmp/egsnrc_sources/${EGSNRC_TARBALL_NAME}" ]; then \
        echo "Using local ${EGSNRC_TARBALL_NAME}"; \
        cp "/tmp/egsnrc_sources/${EGSNRC_TARBALL_NAME}" /tmp/EGSnrc.tar.gz; \
    else \
        echo "Downloading ${EGSNRC_VERSION} from ${EGSNRC_REPO}"; \
        curl -L "${EGSNRC_REPO}/archive/refs/tags/${EGSNRC_VERSION}.tar.gz" -o /tmp/EGSnrc.tar.gz; \
    fi; \
    tar xzf /tmp/EGSnrc.tar.gz -C /opt; \
    mv /opt/EGSnrc-* /opt/EGSnrc

# Bump radial bins for EDKnrc (allow dr=0.5 mm up to r=20 cm => 400 shells)
RUN sed -i 's/REPLACE {$MAXRADII} WITH   {65}/REPLACE {$MAXRADII} WITH   {450}/' /opt/EGSnrc/HEN_HOUSE/user_codes/edknrc/edknrc.mortran && \
    sed -i 's/REPLACE {$MAXRADII+1} WITH {66}/REPLACE {$MAXRADII+1} WITH {451}/' /opt/EGSnrc/HEN_HOUSE/user_codes/edknrc/edknrc.mortran && \
    sed -i 's/REPLACE {$MAXRC} WITH      {65}/REPLACE {$MAXRC} WITH      {450}/' /opt/EGSnrc/HEN_HOUSE/user_codes/edknrc/edknrc.mortran && \
    sed -i 's/REPLACE {$MXREG} WITH      {1153}/REPLACE {$MXREG} WITH      {50000}/' /opt/EGSnrc/HEN_HOUSE/user_codes/edknrc/edknrc.mortran && \
    sed -i 's/REPLACE {$MAXCANGLE} WITH  {48}/REPLACE {$MAXCANGLE} WITH  {64}/' /opt/EGSnrc/HEN_HOUSE/user_codes/edknrc/edknrc.mortran

ENV HEN_HOUSE=/opt/EGSnrc/HEN_HOUSE

# Non-interactive configure (no precompile, config name docker.conf)
RUN cd /opt/EGSnrc && \
    $HEN_HOUSE/scripts/configure.expect docker.conf 3

ENV EGS_HOME=/opt/EGSnrc/egs_home
ENV EGS_CONFIG=$HEN_HOUSE/specs/docker.conf
ENV PATH="$HEN_HOUSE/bin:$EGS_HOME/bin:$PATH"

# Build EDKnrc user code (using docker config)
RUN . $HEN_HOUSE/scripts/egsnrc_bashrc_additions && \
    cd $HEN_HOUSE/user_codes/edknrc && \
    $HEN_HOUSE/scripts/compile_user_code f edknrc config=$EGS_CONFIG

# Add runtime user
RUN useradd -m egs && mkdir -p /scratch /out && chown -R egs:egs /scratch /out
RUN chown -R egs:egs /opt/EGSnrc/egs_home /opt/EGSnrc/HEN_HOUSE
USER egs

# Install helper scripts
WORKDIR /app
COPY egsnrc_edk/scripts /app/scripts
COPY egsnrc_edk/entrypoints /app/
COPY egsnrc_edk/configs /app/configs
COPY egsnrc_edk/examples /app/examples

RUN pip3 install --no-cache-dir numpy pyyaml pandas

ENV PYTHONPATH=/app/scripts
ENV HEN_HOUSE=/opt/EGSnrc/HEN_HOUSE
ENV EGS_HOME=/opt/EGSnrc/egs_home
ENV EGS_CONFIG=$HEN_HOUSE/specs/docker.conf
ENV PATH="$HEN_HOUSE/bin:$EGS_HOME/bin:$PATH"

ENTRYPOINT ["/bin/bash"]
