# Installing pepper dependencies

There are two steps here.

First, you will need a bunch of tools that your distribution almost
certainly packages for you.  See "Tools." If you are using a distribution
other than the ones we've tested, there's a section at the end of this file
that should give you enough information to figure out which ones to install.

Second, you will need to compile several third-party packages that
we distribute. We make a script available that will take care of this
for you; see "Packages," below, for more details.

## Tools

Your distribution will likely have these as binary packages.

Tested with Debian (Wheezy and Jessie), CentOS 7, Fedora 21, Arch 2015.03ish.

### Debian

Note: Ubuntu should be largely similar if not identical.

    apt-get install python2.7 libpython2.7 libpython2.7-dev openjdk-7-jdk \
            golang build-essential gcc g++ gcc-multilib g++-multilib ant  \
            ant-optional make time libboost-all-dev libgmp10 libgmp-dev   \
            zlib1g zlib1g-dev openmpi-bin openmpi-common libopenmpi-dev   \
            libssl-dev

### Fedora

CentOS should be largely similar, assuming a sufficiently up-to-date release.
We have tested this with CentOS 7.

    yum groupinstall 'Development Tools'
    yum install python python-devel java-1.8.0-openjdk-devel golang gcc   \
        gcc-c++ glibc-devel libstdc++ glibc-devel.i686 libstdc++.i686 ant \
        make time boost boost-devel gmp gmp-devel zlib zlib-devel openmpi \
        openmpi-devel openssl-devel

**IMPORTANT**: on Fedora, you need to take a few more steps to use openmpi.
You will need to set up your environment this way every time you want
to run a Pepper computation!

    source /etc/profile.d/modules.sh
    module load mpi/openmpi-x86_64

### Arch

First, you need to enable multilib. See https://wiki.archlinux.org/index.php/multilib

    pacman -S python python2 jdk7-openjdk go base-devel multilib-devel    \
           libstdc++5 lib32-glibc lib32-libstdc++5 lib32-gcc-libs         \
           gcc-libs-multilib libtool-multilib gcc-multilib apache-ant     \
           make time boost boost-libs gmp zlib openmpi

Note that by default Arch uses python3 instead of python2. We haven't tested extensively,
but it seems to work. If you have problems, you can try

    ln -sf /usr/bin/python2 /usr/bin/python
    ln -sf /usr/bin/python2-config /usr/bin/python-config

### Other distributions

For more details, see the end of this file.

## Packages

We have archived known-working versions of these packages at
https://github.com/pepper-project/thirdparty
along with an install script.

Please check out this repo and run the install script after
installing the packages listed in Tools, above, like so:

    git clone https://github.com/pepper-project/thirdparty
    cd thirdparty
    ./install_pepper_deps.sh

When you've finished installing all of these, you should run `sudo ldconfig` for good measure.

## Tool hints for other distributions

### Required packages

1.  python 2.x

2.  java >= 1.7

3.  go >= 1.0.2

4.  gcc, g++ >= 4.7.3 (including multilib support)

5.  Apache ant

6.  GNU Make

7.  GNU time

8.  Boost

9. gmp (including development libraries)

10. zlib

11. MPI

**IMPORTANT**: on Fedora-like OSes, you will need to take a few more steps to use openmpi.
You will need to set up your environment this way every time you want to run a Pepper computation!

    source /etc/profile.d/modules.sh
    module load mpi/openmpi-x86_64

### Optional packages

1.  MySQL (for baseline measurements vs. Pantry database operations)

2.  NVidia CUDA SDK (if you want to use the experimental GPU code, currently undocumented,
and somewhat neglected)

3.  subversion (if you want to check out a later version of clang for use with the BuffetFSM
compiler; this is unsupported)
